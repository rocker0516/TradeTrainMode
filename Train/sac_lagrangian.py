import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from .architectures import Actor, Critic
from .buffer import ReplayBuffer
from .config import Config

class SACLagrangianAgent:
    """
    SAC Agent with Lagrangian Constraint Handling (SAC-Lagrangian).
    Multi-Constraint Support.
    """
    def __init__(
        self,
        price_input_channels: int,
        price_window_size: int,
        state_dim: int,
        action_dim: int,
        cost_limits: List[float], 
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        use_lagrangian: bool = True,
        lagrangian_lr: float = 0.5
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.cost_limits = torch.tensor(cost_limits, device=device, dtype=torch.float32)
        self.num_constraints = len(cost_limits)
        # EMA of cost estimates for stable lambda updates
        self.cost_ema = torch.zeros(self.num_constraints, device=device, dtype=torch.float32)
        self._update_step_count = 0
        
        # --- Actor & Critic ---
        # State Dim + Num Constraints (for lambda injection)
        self.actor_input_dim = state_dim + self.num_constraints
        self.critic_input_dim = state_dim + self.num_constraints
        
        self.actor = Actor(price_input_channels, price_window_size, self.actor_input_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic (Double Q) for Reward
        self.critic_1 = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=1).to(device)
        self.critic_2 = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=1).to(device)
        self.critic_1_target = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=1).to(device)
        self.critic_2_target = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=1).to(device)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)
        
        # --- Entropy / Alpha ---
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=device)

        # --- Lagrangian (Safety) ---
        # Minimizing Cost Constraint: J_Ci(pi) <= di
        self.use_lagrangian = use_lagrangian
        
        if self.use_lagrangian:
            # Cost Critic - Outputs a vector of size num_constraints
            self.cost_critic_1 = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=self.num_constraints).to(device)
            self.cost_critic_1_target = Critic(price_input_channels, price_window_size, self.critic_input_dim, action_dim, output_dim=self.num_constraints).to(device)
            self.cost_critic_1_target.load_state_dict(self.cost_critic_1.state_dict())
            self.cost_critic_optimizer = optim.Adam(self.cost_critic_1.parameters(), lr=lr)
            
            # Lagrangian Multipliers (Lambdas)
            # One lambda per constraint
            self.log_lambda = torch.zeros(self.num_constraints, requires_grad=True, device=device)
            self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lagrangian_lr)
            self.lagrangian_lambda = self.log_lambda.exp()
        else:
            self.lagrangian_lambda = torch.zeros(self.num_constraints, device=device)

    def _augment_state(self, state_vec: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Augments the state vector with current lagrangian multipliers.
        lam: (num_constraints, )
        state_vec: (Batch, state_dim)
        Output: (Batch, state_dim + num_constraints)
        """
        batch_size = state_vec.shape[0]
        # Detach lambda to prevent gradient flow from actor/critic back to lambda optimizer through state
        # However, in select_action it's no_grad anyway.
        # During update, we use the lambda value at that time (or from buffer? No, we should use CURRENT lambda state)
        # Standard practice: Use current lambda for policy conditioning
        
        # Repeat lambda for batch
        lam_batch = lam.detach().unsqueeze(0).repeat(batch_size, 1) # (Batch, num_constraints)
        
        # Normalize lambda for network stability (optional but recommended)
        # Assuming lambda can be large (e.g. 100), log1p or simple scaling might help.
        # Here we pass raw or log1p. Let's use log1p to compress range.
        lam_batch = torch.log1p(lam_batch)
        
        return torch.cat([state_vec, lam_batch], dim=1)

    def set_cost_limits(self, cost_limits: List[float]) -> None:
        """更新當前 cost limits（用於 annealing / curriculum）。"""
        if len(cost_limits) != self.num_constraints:
            raise ValueError(f"cost_limits length mismatch: expected {self.num_constraints}, got {len(cost_limits)}")
        self.cost_limits = torch.tensor(cost_limits, device=self.device, dtype=torch.float32)

    def get_cost_limits(self) -> np.ndarray:
        """取得當前有效 cost limits（for dashboard）。"""
        return self.cost_limits.detach().cpu().numpy()

    def _lambda_gate(self, *, global_step: Optional[int] = None) -> float:
        """
        λ 啟用門檻：
        - 前 warmup：gate=0（λ=0，不更新）
        - ramp：gate 線性從 0 -> 1
        - 之後：gate=1
        """
        step = int(global_step) if global_step is not None else int(self._update_step_count)
        warmup = int(getattr(Config, "LAGRANGIAN_WARMUP_STEPS", 0))
        ramp = int(getattr(Config, "LAGRANGIAN_RAMP_STEPS", 0))
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        frac = (step - warmup) / max(1, ramp)
        return float(min(1.0, max(0.0, frac)))

    def select_action(self, obs: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        # Single observation handling
        # Add batch dimension
        price_seq = torch.FloatTensor(obs['price_seq']).unsqueeze(0).to(self.device)
        
        # Reconstruct state_vector from split observation
        state_parts = [
            obs['account_state'],
            obs['time_state'],
            obs['rhythm_state'],
            obs['cost_state'],
            obs['market_state']
        ]
        state_vec_np = np.concatenate(state_parts)
        state_vec = torch.FloatTensor(state_vec_np).unsqueeze(0).to(self.device)
        
        # Augment State with Lambda
        state_vec = self._augment_state(state_vec, self.lagrangian_lambda)
        
        self.actor.eval()
        with torch.no_grad():
            mean, log_std = self.actor(price_seq, state_vec)
            std = log_std.exp()
            
            if evaluate:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
                
        self.actor.train()
        return action.cpu().numpy()[0]

    def select_action_batch(self, obs: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        """
        Batched action selection.
        Input obs values are expected to be (Batch, ...)
        """
        price_seq = torch.FloatTensor(obs['price_seq']).to(self.device)
        
        # Concatenate state parts along axis 1 (Batch is 0)
        state_parts = [
            obs['account_state'],
            obs['time_state'],
            obs['rhythm_state'],
            obs['cost_state'],
            obs['market_state']
        ]
        # Check if they are numpy arrays
        state_vec_np = np.concatenate(state_parts, axis=1)
        state_vec = torch.FloatTensor(state_vec_np).to(self.device)
        
        # Augment State
        state_vec = self._augment_state(state_vec, self.lagrangian_lambda)
        
        self.actor.eval()
        with torch.no_grad():
            mean, log_std = self.actor(price_seq, state_vec)
            std = log_std.exp()
            
            if evaluate:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
                
        self.actor.train()
        return action.cpu().numpy()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int, *, global_step: Optional[int] = None) -> Dict[str, float]:
        batch = replay_buffer.sample(batch_size)
        self._update_step_count += 1
        
        price_seq = batch['price_seq']
        state_vec = batch['state_vec']
        action = batch['action']
        reward = batch['reward']
        cost = batch['cost'] # (B, num_constraints)
        next_price_seq = batch['next_price_seq']
        next_state_vec = batch['next_state_vec']
        done = batch['done']
        
        # Augment States with CURRENT Lambda
        # We use current lambda for both current and next state evaluation
        # This makes the policy conditioned on "current safety urgency"
        curr_lam = self.lagrangian_lambda.detach() # (num_constraints, )
        state_vec_aug = self._augment_state(state_vec, curr_lam)
        next_state_vec_aug = self._augment_state(next_state_vec, curr_lam)

        # 1. Update Critic (Reward Q-Functions)
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_price_seq, next_state_vec_aug)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = next_dist.rsample()
            next_action = torch.tanh(next_action_sample)
            
            next_log_prob = next_dist.log_prob(next_action_sample) - torch.log(1 - next_action.pow(2) + 1e-6)
            next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
            
            target_q1 = self.critic_1_target(next_price_seq, next_state_vec_aug, next_action)
            target_q2 = self.critic_2_target(next_price_seq, next_state_vec_aug, next_action)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            target_q_value = reward + (1 - done) * self.gamma * min_target_q

        current_q1 = self.critic_1(price_seq, state_vec_aug, action)
        current_q2 = self.critic_2(price_seq, state_vec_aug, action)
        
        critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. Update Cost Critic (if using Lagrangian)
        if self.use_lagrangian:
            with torch.no_grad():
                target_qc1 = self.cost_critic_1_target(next_price_seq, next_state_vec_aug, next_action)
                target_qc_value = cost + (1 - done) * self.gamma * target_qc1
            
            current_qc1 = self.cost_critic_1(price_seq, state_vec_aug, action)
            cost_critic_loss = F.mse_loss(current_qc1, target_qc_value)
            
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()

        # 3. Update Actor
        mean, log_std = self.actor(price_seq, state_vec_aug)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        current_action = torch.tanh(action_sample)
        
        log_prob = dist.log_prob(action_sample) - torch.log(1 - current_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        q1_pi = self.critic_1(price_seq, state_vec_aug, current_action)
        q2_pi = self.critic_2(price_seq, state_vec_aug, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_prob) - min_q_pi
        
        if self.use_lagrangian:
            qc_pi = self.cost_critic_1(price_seq, state_vec_aug, current_action)
            
            gate = self._lambda_gate(global_step=global_step)
            lam_base = self.log_lambda.exp().detach()  # base λ (>=0)
            lam = lam_base * gate  # effective λ
            self.lagrangian_lambda = lam  # for logging/conditioning
            # Sum over constraints: sum(lambda_i * Q_Ci)
            # qc_pi is (Batch, num_constraints)
            # lam is (num_constraints) -> broadcast
            lagrangian_penalty = (qc_pi * lam).sum(dim=1, keepdim=True)
            
            actor_loss = actor_loss + lagrangian_penalty
            
            current_cost_estimate = qc_pi.mean(dim=0) # (num_constraints,)

        actor_loss = actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 4. Update Alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
        # 5. Update Lambda
        if self.use_lagrangian:
            gate = self._lambda_gate(global_step=global_step)
            if gate <= 0.0:
                # warm-up：不更新 λ，並保持 effective λ = 0
                lambda_loss = torch.tensor(0.0, device=self.device)
                violation = torch.zeros_like(qc_pi, device=self.device)
                self.lagrangian_lambda = torch.zeros(self.num_constraints, device=self.device)
            else:
                # --- EMA of cost estimate for stable λ updates ---
                beta = float(getattr(Config, "LAGRANGIAN_COST_EMA_BETA", 0.99))
                beta = float(np.clip(beta, 0.0, 0.9999))
                with torch.no_grad():
                    self.cost_ema = beta * self.cost_ema + (1.0 - beta) * current_cost_estimate.detach()

                # Violation uses EMA (not per-batch noise)
                violation_vec = self.cost_ema - self.cost_limits
                # Broadcast to (B, num_constraints) for logging compatibility
                violation = violation_vec.unsqueeze(0).repeat(qc_pi.shape[0], 1)

                # Use exp(log_lambda) to keep lambda >= 0 and apply correct ascent direction
                lambda_vals = self.log_lambda.exp() * gate  # ramp affects update strength too
                # Loss to minimize: - lambda * violation (gradient ascent on lambda)
                lambda_loss = - (lambda_vals * violation_vec).mean()

                self.lambda_optimizer.zero_grad()
                lambda_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.log_lambda], 1.0)
                self.lambda_optimizer.step()
            
            with torch.no_grad():
                # Clamp Lambda (log-space) for numerical stability only
                lam_min = float(getattr(Config, "LAMBDA_LOG_CLAMP_MIN", -5.0))
                lam_max = float(getattr(Config, "LAMBDA_LOG_CLAMP_MAX", 5.0))
                if lam_max < lam_min:
                    lam_max = lam_min
                self.log_lambda.data.clamp_(min=lam_min, max=lam_max)
            # effective λ already handled by gate above

        # 6. Soft Updates
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)
        if self.use_lagrangian:
            self._soft_update(self.cost_critic_1, self.cost_critic_1_target)

        # Metrics
        metrics = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "val/alpha": self.alpha.item(),
            "val/avg_q": min_q_pi.mean().item(),
        }
        
        if self.use_lagrangian:
            metrics["loss/cost_critic"] = cost_critic_loss.item()
            metrics["loss/lambda"] = lambda_loss.item()
            for i in range(self.num_constraints):
                metrics[f"val/lambda_{i+1}"] = self.lagrangian_lambda[i].item()
                metrics[f"val/avg_cost_q_{i+1}"] = current_cost_estimate[i].item()
                metrics[f"val/cost_violation_{i+1}"] = violation[:, i].mean().item()
                # EMA-based diagnostics (the values actually driving lambda updates)
                metrics[f"val/cost_ema_{i+1}"] = self.cost_ema[i].detach().item()
                # violation is broadcasted for logging; use vector form for clarity
                metrics[f"val/cost_violation_ema_{i+1}"] = (self.cost_ema[i] - self.cost_limits[i]).detach().item()
            metrics["val/lambda_gate"] = float(self._lambda_gate(global_step=global_step))
            
        return metrics

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
