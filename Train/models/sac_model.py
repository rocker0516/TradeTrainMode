"""
SAC (Soft Actor-Critic) 模型實作

純 SAC 實作，使用多層全連接網絡處理觀察。
遵循單一職責原則 (SRP)，每個網絡類別只負責自己的功能。
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from pathlib import Path

from .base_model import BaseRLModel


# 常數定義
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class Actor(nn.Module):
    """
    Actor 網絡（策略網絡）
    
    輸出動作分佈的均值和標準差。
    
    Args:
        state_dim: 狀態維度
        action_dim: 動作維度
        hidden_dim: 隱藏層維度
        num_layers: 網絡層數
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ) -> None:
        super(Actor, self).__init__()
        
        # 構建多層網絡
        layers = []
        input_dim = state_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            state: 狀態張量
            
        Returns:
            (mean, log_std): 動作分佈的均值和對數標準差
        """
        x = self.shared_net(state)
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        evaluate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        採樣動作
        
        Args:
            state: 狀態張量
            evaluate: 是否為評估模式（使用均值，無隨機性）
            
        Returns:
            (action, log_prob, mean): 動作、對數機率、均值
        """
        mean, log_std = self.forward(state)
        
        if evaluate:
            # 評估模式：使用均值
            action = torch.tanh(mean)
            return action, torch.zeros_like(action), mean
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重參數化技巧
        action = torch.tanh(x_t)
        
        # 計算對數機率（考慮 tanh 變換）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        
        return action, log_prob, mean


class Critic(nn.Module):
    """
    Critic 網絡（Q 網絡）
    
    評估狀態-動作對的價值。
    
    Args:
        state_dim: 狀態維度
        action_dim: 動作維度
        hidden_dim: 隱藏層維度
        num_layers: 網絡層數
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ) -> None:
        super(Critic, self).__init__()
        
        # Q1 網絡
        q1_layers = []
        input_dim = state_dim + action_dim
        for _ in range(num_layers):
            q1_layers.append(nn.Linear(input_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            input_dim = hidden_dim
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1_net = nn.Sequential(*q1_layers)
        
        # Q2 網絡（雙 Q 網絡減少過估計）
        q2_layers = []
        input_dim = state_dim + action_dim
        for _ in range(num_layers):
            q2_layers.append(nn.Linear(input_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            input_dim = hidden_dim
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(*q2_layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            state: 狀態張量
            action: 動作張量
            
        Returns:
            (q1, q2): 兩個 Q 值
        """
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        
        return q1, q2


class SACModel(BaseRLModel):
    """
    SAC 模型（純 SAC，不使用 LSTM）
    
    使用多層全連接網絡處理扁平化的觀察空間。
    
    Args:
        observation_shape: 觀察空間形狀
        action_dim: 動作維度
        hidden_dim: 隱藏層維度
        num_layers: 網絡層數
        lr: 學習率
        alpha: 熵係數
        gamma: 折扣因子
        tau: 軟更新係數
        auto_entropy_tuning: 是否自動調整熵係數
        device: 訓練設備
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        lr: float = 3e-4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_entropy_tuning: bool = True,
        device: str = 'cuda'
    ) -> None:
        super().__init__(observation_shape, action_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 計算狀態維度（扁平化）
        state_dim = int(np.prod(observation_shape))
        
        # Actor
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        # Critic
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        # Target Critic（目標網絡）
        self.critic_target = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        # 複製參數到目標網絡
        self._hard_update(self.critic_target, self.critic)
        
        # 優化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 自動熵調整
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor([action_dim]).to(device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> np.ndarray:
        """
        選擇動作
        
        Args:
            state: 當前狀態
            evaluate: 是否為評估模式
            
        Returns:
            選擇的動作
        """
        # 扁平化狀態
        state_flat = state.flatten()
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.actor.sample(state_tensor, evaluate=evaluate)
        
        return action.cpu().numpy().flatten()
    
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        更新模型參數
        
        Args:
            batch: 批次數據
            
        Returns:
            損失字典
        """
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # 扁平化狀態
        batch_size = state.shape[0]
        state_flat = state.view(batch_size, -1)
        next_state_flat = next_state.view(batch_size, -1)
        
        # === 更新 Critic ===
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_flat)
            q1_next, q2_next = self.critic_target(next_state_flat, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * min_q_next
        
        q1, q2 = self.critic(state_flat, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === 更新 Actor ===
        new_action, log_prob, _ = self.actor.sample(state_flat)
        q1_new, q2_new = self.critic(state_flat, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - min_q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # === 更新 Alpha（熵係數） ===
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # === 軟更新目標網絡 ===
        self._soft_update(self.critic_target, self.critic)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'q_value': min_q_new.mean().item(),
        }
    
    def _soft_update(
        self,
        target: nn.Module,
        source: nn.Module
    ) -> None:
        """
        軟更新目標網絡
        
        Args:
            target: 目標網絡
            source: 源網絡
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def _hard_update(
        self,
        target: nn.Module,
        source: nn.Module
    ) -> None:
        """
        硬更新目標網絡（直接複製）
        
        Args:
            target: 目標網絡
            source: 源網絡
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路徑
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, save_path)
    
    def load(self, path: str) -> None:
        """
        加載模型
        
        Args:
            path: 模型路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
    
    def get_state_dict(self) -> Dict[str, Any]:
        """獲取模型狀態字典"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加載模型狀態字典"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
    
    def train(self) -> None:
        """設置為訓練模式"""
        super().train()
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
    
    def eval(self) -> None:
        """設置為評估模式"""
        super().eval()
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

