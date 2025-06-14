import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
import numpy as np
from typing import Tuple, Dict

class Actor(nn.Module):
    """
    Actor網絡 - 策略網絡
    輸出連續動作的均值和標準差
    動作空間：[交易方向(-1~1), 止盈比例(0.2~10), 止損比例(0.1~0.3)]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 配置參數
        input_dim = config.TRANSFORMER_CONFIG['embed_dim']
        hidden_dim = config.ACTOR_CRITIC_CONFIG['hidden_dim']
        num_layers = config.ACTOR_CRITIC_CONFIG['num_layers']
        dropout = config.ACTOR_CRITIC_CONFIG['dropout']
        action_dim = config.ACTOR_CRITIC_CONFIG['action_dim']
        self.action_bounds = config.ACTOR_CRITIC_CONFIG['action_bounds']
        
        # 共享特徵提取層
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 動作分支 - 分別處理不同類型的動作
        # 1. 交易方向分支 (-1 to 1)
        self.direction_mean = nn.Linear(hidden_dim, 1)
        self.direction_std = nn.Linear(hidden_dim, 1)
        
        # 2. 止盈比例分支 (0.2 to 10)
        self.take_profit_alpha = nn.Linear(hidden_dim, 1)
        self.take_profit_beta = nn.Linear(hidden_dim, 1)
        
        # 3. 止損比例分支 (0.1 to 0.3)
        self.stop_loss_alpha = nn.Linear(hidden_dim, 1)
        self.stop_loss_beta = nn.Linear(hidden_dim, 1)
        
        # 初始化參數
        self._init_weights()
    
    def _init_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        Args:
            state: [batch_size, 1, embed_dim]
        Returns:
            action_means: [batch_size, action_dim]
            action_stds: [batch_size, action_dim]
        """
        # 展平輸入
        if len(state.shape) == 3:
            state = state.squeeze(1)  # [batch_size, embed_dim]
        
        # 共享特徵提取
        shared_features = self.shared_layers(state)
        
        # 1. 交易方向 - 使用 tanh 激活函數限制在 [-1, 1]
        direction_mean = torch.tanh(self.direction_mean(shared_features))
        direction_std = F.softplus(self.direction_std(shared_features)) + 1e-5
        
        # 2. 止盈比例 - 使用 Beta 分佈參數化
        take_profit_alpha = F.softplus(self.take_profit_alpha(shared_features)) + 1.0
        take_profit_beta = F.softplus(self.take_profit_beta(shared_features)) + 1.0
        
        # 3. 止損比例 - 使用 Beta 分佈參數化
        stop_loss_alpha = F.softplus(self.stop_loss_alpha(shared_features)) + 1.0
        stop_loss_beta = F.softplus(self.stop_loss_beta(shared_features)) + 1.0
        
        return {
            'direction': (direction_mean, direction_std),
            'take_profit': (take_profit_alpha, take_profit_beta),
            'stop_loss': (stop_loss_alpha, stop_loss_beta)
        }
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        採樣動作
        Returns:
            actions: [batch_size, action_dim]
            log_probs: [batch_size, action_dim]
        """
        action_params = self.forward(state)
        
        # 1. 交易方向 - Normal distribution
        direction_mean, direction_std = action_params['direction']
        direction_dist = Normal(direction_mean, direction_std)
        direction_action = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction_action)
        
        # 2. 止盈比例 - Beta distribution scaled to [0.2, 10]
        tp_alpha, tp_beta = action_params['take_profit']
        tp_dist = Beta(tp_alpha, tp_beta)
        tp_sample = tp_dist.sample()
        tp_action = tp_sample * (10.0 - 0.2) + 0.2  # scale to [0.2, 10]
        tp_log_prob = tp_dist.log_prob(tp_sample)
        
        # 3. 止損比例 - Beta distribution scaled to [0.1, 0.3]
        sl_alpha, sl_beta = action_params['stop_loss']
        sl_dist = Beta(sl_alpha, sl_beta)
        sl_sample = sl_dist.sample()
        sl_action = sl_sample * (0.3 - 0.1) + 0.1  # scale to [0.1, 0.3]
        sl_log_prob = sl_dist.log_prob(sl_sample)
        
        # 組合動作和對數概率
        actions = torch.cat([direction_action, tp_action, sl_action], dim=-1)
        log_probs = torch.cat([direction_log_prob, tp_log_prob, sl_log_prob], dim=-1)
        
        return actions, log_probs
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        評估動作的對數概率和熵
        Args:
            state: [batch_size, embed_dim]
            action: [batch_size, action_dim]
        Returns:
            log_probs: [batch_size, action_dim]
            entropy: [batch_size, action_dim]
        """
        action_params = self.forward(state)
        
        # 分解動作
        direction_action = action[:, 0:1]
        tp_action = action[:, 1:2]
        sl_action = action[:, 2:3]
        
        # 1. 交易方向
        direction_mean, direction_std = action_params['direction']
        direction_dist = Normal(direction_mean, direction_std)
        direction_log_prob = direction_dist.log_prob(direction_action)
        direction_entropy = direction_dist.entropy()
        
        # 2. 止盈比例 - 需要反向縮放
        tp_alpha, tp_beta = action_params['take_profit']
        tp_scaled = (tp_action - 0.2) / (10.0 - 0.2)  # scale back to [0, 1]
        tp_scaled = torch.clamp(tp_scaled, 1e-6, 1-1e-6)  # 避免邊界值
        tp_dist = Beta(tp_alpha, tp_beta)
        tp_log_prob = tp_dist.log_prob(tp_scaled)
        tp_entropy = tp_dist.entropy()
        
        # 3. 止損比例 - 需要反向縮放
        sl_alpha, sl_beta = action_params['stop_loss']
        sl_scaled = (sl_action - 0.1) / (0.3 - 0.1)  # scale back to [0, 1]
        sl_scaled = torch.clamp(sl_scaled, 1e-6, 1-1e-6)  # 避免邊界值
        sl_dist = Beta(sl_alpha, sl_beta)
        sl_log_prob = sl_dist.log_prob(sl_scaled)
        sl_entropy = sl_dist.entropy()
        
        # 組合結果
        log_probs = torch.cat([direction_log_prob, tp_log_prob, sl_log_prob], dim=-1)
        entropy = torch.cat([direction_entropy, tp_entropy, sl_entropy], dim=-1)
        
        return log_probs, entropy

class Critic(nn.Module):
    """
    Critic網絡 - 價值網絡
    估計狀態價值 V(s)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 配置參數
        input_dim = config.TRANSFORMER_CONFIG['embed_dim']
        hidden_dim = config.ACTOR_CRITIC_CONFIG['hidden_dim']
        num_layers = config.ACTOR_CRITIC_CONFIG['num_layers']
        dropout = config.ACTOR_CRITIC_CONFIG['dropout']
        
        # 價值網絡
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))  # 輸出標量價值
        
        self.value_network = nn.Sequential(*layers)
        
        # 初始化參數
        self._init_weights()
    
    def _init_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        Args:
            state: [batch_size, 1, embed_dim]
        Returns:
            value: [batch_size, 1]
        """
        # 展平輸入
        if len(state.shape) == 3:
            state = state.squeeze(1)  # [batch_size, embed_dim]
        
        value = self.value_network(state)
        return value

class ActorCritic(nn.Module):
    """
    Actor-Critic網絡
    結合策略網絡和價值網絡
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.actor = Actor(config)
        self.critic = Critic(config)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        Args:
            state: [batch_size, 1, embed_dim]
        Returns:
            actions: [batch_size, action_dim]
            log_probs: [batch_size, action_dim]
            values: [batch_size, 1]
        """
        # 採樣動作
        actions, log_probs = self.actor.sample_action(state)
        
        # 估計狀態價值
        values = self.critic(state)
        
        return actions, log_probs, values
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        評估狀態-動作對
        Args:
            state: [batch_size, embed_dim]
            action: [batch_size, action_dim]
        Returns:
            log_probs: [batch_size, action_dim]
            values: [batch_size, 1]
            entropy: [batch_size, action_dim]
        """
        # 評估動作
        log_probs, entropy = self.actor.evaluate_action(state, action)
        
        # 估計狀態價值
        values = self.critic(state)
        
        return log_probs, values, entropy
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        獲取動作（用於推理）
        Args:
            state: [1, 1, embed_dim]
            deterministic: 是否使用確定性策略
        Returns:
            action: [action_dim]
        """
        with torch.no_grad():
            if deterministic:
                # 確定性策略：使用均值
                action_params = self.actor.forward(state)
                
                # 交易方向：使用均值
                direction = action_params['direction'][0]
                
                # 止盈比例：使用Beta分佈的期望值
                tp_alpha, tp_beta = action_params['take_profit']
                tp_mean = tp_alpha / (tp_alpha + tp_beta)
                tp_action = tp_mean * (10.0 - 0.2) + 0.2
                
                # 止損比例：使用Beta分佈的期望值
                sl_alpha, sl_beta = action_params['stop_loss']
                sl_mean = sl_alpha / (sl_alpha + sl_beta)
                sl_action = sl_mean * (0.3 - 0.1) + 0.1
                
                action = torch.cat([direction, tp_action, sl_action], dim=-1)
            else:
                # 隨機策略：採樣
                action, _ = self.actor.sample_action(state)
            
            return action.cpu().numpy().flatten()
    
    def get_value(self, state: torch.Tensor) -> float:
        """
        獲取狀態價值（用於推理）
        Args:
            state: [1, 1, embed_dim]
        Returns:
            value: scalar
        """
        with torch.no_grad():
            value = self.critic(state)
            return value.item() 