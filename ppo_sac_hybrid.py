import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(FeatureExtractor, self).__init__()
        # input_dim: n_features
        self.input_projection = nn.Linear(input_dim, 27)
        self.lstm = nn.LSTM(
            input_size=27,  # LSTM expects 27 features per time step
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_norm = nn.LayerNorm(27)

    def forward(self, x):
        # x: [batch, n_features, window_size]
        if torch.isnan(x).any():
            print("Warning: NaN values in input to FeatureExtractor")
            x = torch.nan_to_num(x, nan=0.0)
        # 1. 轉換為 [batch, window_size, n_features]
        x = x.permute(0, 2, 1)  # [batch, window_size, n_features]
        # 2. 投影到 27 維
        x = self.input_projection(x)  # [batch, window_size, 27]
        # 3. 輸入標準化
        x = self.input_norm(x)
        # 4. 檢查標準化後的數值
        if torch.isnan(x).any():
            print("Warning: NaN values after input normalization")
            x = torch.nan_to_num(x, nan=0.0)
        # 5. LSTM 前向傳播
        lstm_out, _ = self.lstm(x)
        # 6. 檢查 LSTM 輸出
        if torch.isnan(lstm_out).any():
            print("Warning: NaN values in LSTM output")
            lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        # 7. 應用層標準化
        lstm_out = self.layer_norm(lstm_out)
        # 8. 最終檢查
        if torch.isnan(lstm_out).any():
            print("Warning: NaN values after layer normalization")
            lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        return lstm_out[:, -1, :]

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # 使用 tanh 限制均值範圍在 [-1, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制 log_std 的範圍
        return mean, log_std

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim=None, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim + (action_dim if action_dim else 0), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, a=None):
        if a is not None:
            x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)

class PPOSACHybrid:
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.action_dim = action_dim
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=state_dim,  # 直接使用 state_dim 作為特徵數量
            hidden_dim=256
        ).to(device)
        
        # PPO 組件
        self.ppo_actor = ActorNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        self.ppo_critic = CriticNetwork(
            input_dim=256
        ).to(device)
        
        # SAC 組件
        self.sac_actor = ActorNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        self.sac_critic1 = CriticNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        self.sac_critic2 = CriticNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        
        # 目標網絡
        self.target_critic1 = CriticNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        self.target_critic2 = CriticNetwork(
            input_dim=256,
            action_dim=action_dim
        ).to(device)
        
        # 初始化目標網絡
        self.target_critic1.load_state_dict(self.sac_critic1.state_dict())
        self.target_critic2.load_state_dict(self.sac_critic2.state_dict())
        
        # 分開的優化器
        # PPO 優化器
        self.ppo_actor_optimizer = torch.optim.Adam(
            list(self.ppo_actor.parameters()),
            lr=1e-4,  # 降低學習率
            eps=1e-5  # 增加 epsilon 值
        )
        self.ppo_critic_optimizer = torch.optim.Adam(
            list(self.ppo_critic.parameters()),
            lr=1e-4,
            eps=1e-5
        )
        
        # SAC 優化器
        self.sac_actor_optimizer = torch.optim.Adam(
            list(self.sac_actor.parameters()),
            lr=1e-4,
            eps=1e-5
        )
        self.sac_critic_optimizer = torch.optim.Adam(
            list(self.sac_critic1.parameters()) + 
            list(self.sac_critic2.parameters()),
            lr=1e-4,
            eps=1e-5
        )
        
        # 特徵提取器優化器
        self.feature_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=1e-4,
            eps=1e-5
        )
        
        # SAC 超參數
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        # PPO 超參數
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # 添加梯度裁剪閾值
        self.max_grad_norm = 1.0
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.feature_extractor(state)
            
            # 獲取 PPO 動作
            ppo_mean, ppo_log_std = self.ppo_actor(features)
            ppo_std = torch.exp(ppo_log_std)
            
            # 檢查並處理 NaN 值
            if torch.isnan(ppo_mean).any() or torch.isnan(ppo_std).any():
                print("Warning: NaN values detected in PPO action distribution")
                ppo_mean = torch.zeros_like(ppo_mean)
                ppo_std = torch.ones_like(ppo_std)
            
            ppo_dist = Normal(ppo_mean, ppo_std)
            ppo_action = ppo_mean if evaluate else ppo_dist.sample()
            
            # 獲取 SAC 動作
            sac_mean, sac_log_std = self.sac_actor(features)
            sac_std = torch.exp(sac_log_std)
            
            # 檢查並處理 NaN 值
            if torch.isnan(sac_mean).any() or torch.isnan(sac_std).any():
                print("Warning: NaN values detected in SAC action distribution")
                sac_mean = torch.zeros_like(sac_mean)
                sac_std = torch.ones_like(sac_std)
            
            sac_dist = Normal(sac_mean, sac_std)
            sac_action = sac_mean if evaluate else sac_dist.sample()
            
            # 根據市場條件選擇動作
            # 這裡可以添加您的市場條件判斷邏輯
            # 暫時使用簡單的隨機選擇
            if np.random.random() < 0.5:
                return ppo_action.cpu().numpy()[0]
            else:
                return sac_action.cpu().numpy()[0]
    
    def update(self, batch):
        # 將數據轉換為張量
        state = torch.FloatTensor(batch['state']).to(self.device)
        action = torch.FloatTensor(batch['action']).to(self.device)
        reward = torch.FloatTensor(batch['reward']).to(self.device)
        next_state = torch.FloatTensor(batch['next_state']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)
        
        # 更新 PPO
        ppo_loss = self._update_ppo(state, action, reward, next_state, done)
        
        # 更新 SAC
        sac_loss = self._update_sac(state, action, reward, next_state, done)
        
        return ppo_loss + sac_loss
    
    def _update_ppo(self, state, action, reward, next_state, done):
        # 提取特徵
        features = self.feature_extractor(state)
        next_features = self.feature_extractor(next_state)
        
        # 計算當前策略的動作概率
        mean, log_std = self.ppo_actor(features)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 計算價值估計
        value = self.ppo_critic(features).squeeze(-1)
        next_value = self.ppo_critic(next_features).squeeze(-1)
        
        # 計算優勢函數
        advantage = reward + (1 - done) * 0.99 * next_value - value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # 計算 PPO 目標
        ratio = torch.exp(log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 計算價值損失
        value_target = reward + (1 - done) * 0.99 * next_value
        value_loss = F.mse_loss(value, value_target.detach())
        
        # 計算熵損失
        entropy_loss = -dist.entropy().mean()
        
        # 計算總損失
        total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # 更新網絡
        self.ppo_actor_optimizer.zero_grad()
        self.ppo_critic_optimizer.zero_grad()
        total_loss.backward()
        
        # 在反向傳播後添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.ppo_actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.ppo_critic.parameters(), self.max_grad_norm)
        self.ppo_actor_optimizer.step()
        self.ppo_critic_optimizer.step()
        
        return total_loss.item()
    
    def _update_sac(self, state, action, reward, next_state, done):
        # 提取特徵
        features = self.feature_extractor(state)
        next_features = self.feature_extractor(next_state)
        
        # 1. 計算目標 Q 值
        with torch.no_grad():
            next_mean, next_log_std = self.sac_actor(next_features)
            next_std = torch.exp(next_log_std)
            next_dist = Normal(next_mean, next_std)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1)
            
            target_q1 = self.target_critic1(next_features, next_action)
            target_q2 = self.target_critic2(next_features, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * 0.99 * (target_q - self.alpha * next_log_prob.unsqueeze(-1))
        
        # 2. 計算當前 Q 值
        current_q1 = self.sac_critic1(features, action)
        current_q2 = self.sac_critic2(features, action)
        
        # 3. 計算 Actor 輸出
        mean, log_std = self.sac_actor(features)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action_sample = dist.rsample()
        log_prob = dist.log_prob(action_sample).sum(dim=-1)
        
        # 4. 計算 Q 值
        q1 = self.sac_critic1(features, action_sample)
        q2 = self.sac_critic2(features, action_sample)
        q = torch.min(q1, q2)
        
        # 5. 計算所有損失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q).mean()
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        
        # 6. 計算總損失
        total_loss = critic_loss + actor_loss + alpha_loss
        
        # 7. 更新所有網絡
        self.sac_critic_optimizer.zero_grad()
        self.sac_actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        
        total_loss.backward()
        
        self.sac_critic_optimizer.step()
        self.sac_actor_optimizer.step()
        self.alpha_optimizer.step()
        
        # 8. 更新目標網絡
        self._update_target_network()
        
        return total_loss.item()
    
    def _update_target_network(self, tau=0.005):
        # 軟更新目標網絡
        for target_param, param in zip(self.target_critic1.parameters(), self.sac_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.sac_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data) 