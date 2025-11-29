import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class PriceEncoder(nn.Module):
    """
    Encoder for Price Sequence using 1D CNN.
    
    Args:
        input_channels (int): Number of input features per time step.
        window_size (int): Length of the time sequence.
        output_dim (int): Dimension of the output embedding.
    """
    def __init__(self, input_channels: int, window_size: int, output_dim: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.window_size = window_size
        
        # 1D CNN Architecture
        # Input: (Batch, Channels, Length) - Note: PyTorch Conv1d expects (B, C, L)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Compute flattened size after convolutions if not using global pool
        # But global average pooling is flexible and robust.
        
        self.fc = nn.Linear(64, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, Window, Features) or (Batch, Features, Window).
               If (Batch, Window, Features), it will be permuted.
        """
        # Ensure input is (B, C, L)
        if x.dim() == 3 and x.shape[1] == self.window_size:
            x = x.permute(0, 2, 1) # (B, W, F) -> (B, F, W)
            
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        
        # Global Average Pooling
        x = self.global_avg_pool(x) # (B, 64, 1)
        x = x.flatten(1)            # (B, 64)
        
        x = self.fc(x)
        x = self.layer_norm(x)
        # Safe ReLU (though LayerNorm usually keeps things sane)
        return F.relu(x)


class Actor(nn.Module):
    """
    SAC Actor Network (Gaussian Policy).
    
    Takes price_seq and state_vector, fuses them, and outputs mean and log_std for actions.
    """
    def __init__(
        self, 
        price_input_channels: int, 
        price_window_size: int, 
        state_dim: int, 
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Encoders
        self.price_encoder = PriceEncoder(price_input_channels, price_window_size, output_dim=128)
        
        # Fusion and Policy Head
        # Combined dim = 128 (price) + state_dim
        fusion_input_dim = 128 + state_dim
        
        self.trunk = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, price_seq: torch.Tensor, state_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        price_emb = self.price_encoder(price_seq)
        
        # Concatenate
        x = torch.cat([price_emb, state_vec], dim=1)
        x = self.trunk(x)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class Critic(nn.Module):
    """
    SAC Critic Network (Q-function).
    
    Takes price_seq, state_vector, and action, and outputs Q-value.
    Uses Double Q-learning (two critics) typically, but this class defines ONE critic.
    The agent will instantiate two of these.
    """
    def __init__(
        self, 
        price_input_channels: int, 
        price_window_size: int, 
        state_dim: int, 
        action_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1
    ):
        super().__init__()
        
        # We can share the price encoder or have separate ones. 
        # For simplicity and stability, separate encoders are often used in RL to avoid interference,
        # or shared if using a larger backbone. Here we use separate for simplicity.
        self.price_encoder = PriceEncoder(price_input_channels, price_window_size, output_dim=128)
        
        # Fusion
        # Input: 128 (price) + state_dim + action_dim
        fusion_input_dim = 128 + state_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, price_seq: torch.Tensor, state_vec: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        price_emb = self.price_encoder(price_seq)
        
        x = torch.cat([price_emb, state_vec, action], dim=1)
        q_value = self.net(x)
        
        return q_value

