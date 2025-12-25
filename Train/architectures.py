import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block to adaptively recalibrate channel-wise feature responses.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class PriceEncoder(nn.Module):
    """
    Encoder for Price Sequence using Deep 1D CNN with SE Blocks.
    Optimized for long sequence (1440 steps).
    """
    def __init__(self, input_channels: int, window_size: int, output_dim: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.window_size = window_size
        
        # Deep 1D CNN Architecture (5 layers)
        # Input: (Batch, Channels, Length)
        
        # Layer 1: Capture very short-term patterns (5-min level)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 1440 -> 720
        
        # Layer 2: Capture short-term patterns (15-30 min)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 720 -> 360
        
        # Layer 3: Capture mid-term patterns (1-2 hour)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.se3 = SEBlock(128) # Attention on channels
        self.pool3 = nn.MaxPool1d(kernel_size=2) # 360 -> 180
        
        # Layer 4: Capture long-term patterns (4-8 hour)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2) # 180 -> 90
        
        # Layer 5: Capture global trends (Daily)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.se5 = SEBlock(256)
        # Global Pooling follows
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(256, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (Batch, Window, Features) or (Batch, Features, Window).
        """
        # Ensure input is (B, C, L)
        if x.dim() == 3 and x.shape[1] == self.window_size:
            x = x.permute(0, 2, 1) # (B, W, F) -> (B, F, W)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x) # Apply SE
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se5(x) # Apply SE
        
        # Global Average Pooling
        x = self.global_avg_pool(x) # (B, 256, 1)
        x = x.flatten(1)            # (B, 256)
        
        x = self.fc(x)
        x = self.layer_norm(x)
        return F.relu(x)

class DailyMacroEncoder(nn.Module):
    """
    Encoder for Daily Macro Sequence using 1D CNN.
    """
    def __init__(self, input_channels: int, window_size: int, output_dim: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.window_size = window_size
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(32, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) or (B, C, W)
        if x.dim() == 3 and x.shape[1] == self.window_size:
            x = x.permute(0, 2, 1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = F.relu(self.ln(self.fc(x)))
        return x

class Actor(nn.Module):
    """
    SAC Actor Network (Gaussian Policy).
    
    Takes price_seq, daily_seq, and state_vector, fuses them, and outputs mean and log_std for actions.
    """
    def __init__(
        self, 
        price_input_channels: int, 
        price_window_size: int,
        state_dim: int, 
        action_dim: int,
        daily_input_channels: int = 1,
        daily_window_size: int = 1,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Encoders
        self.price_encoder = PriceEncoder(price_input_channels, price_window_size, output_dim=128)
        self.daily_input_channels = int(daily_input_channels)
        self.daily_window_size = int(daily_window_size)
        self.daily_encoder = DailyMacroEncoder(max(1, self.daily_input_channels), max(1, self.daily_window_size), output_dim=32)
        
        # Fusion and Policy Head
        # Combined dim = 128 (price) + 32 (daily) + state_dim
        fusion_input_dim = 128 + 32 + state_dim
        
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

    def forward(
        self,
        price_seq: torch.Tensor,
        daily_seq: torch.Tensor | None = None,
        state_vec: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Backward-compat:
        - 舊測試/舊呼叫可能是 actor(price_seq, state_vec)
        - 新版是 actor(price_seq, daily_seq, state_vec)
        """
        # Backward compatible arg routing
        if state_vec is None:
            state_vec = daily_seq  # type: ignore[assignment]
            daily_seq = None

        if state_vec is None:
            raise ValueError("state_vec must be provided")

        price_emb = self.price_encoder(price_seq)
        if daily_seq is None:
            # If daily_seq is missing, feed zeros (keeps net shape stable).
            b = price_seq.shape[0]
            daily_seq = torch.zeros(
                (b, max(1, self.daily_window_size), max(1, self.daily_input_channels)),
                dtype=price_seq.dtype,
                device=price_seq.device,
            )
        daily_emb = self.daily_encoder(daily_seq)
        
        # Concatenate
        x = torch.cat([price_emb, daily_emb, state_vec], dim=1)
        x = self.trunk(x)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class Critic(nn.Module):
    """
    SAC Critic Network (Q-function).
    
    Takes price_seq, daily_seq, state_vector, and action, and outputs Q-value.
    """
    def __init__(
        self, 
        price_input_channels: int, 
        price_window_size: int, 
        state_dim: int, 
        action_dim: int,
        daily_input_channels: int = 1,
        daily_window_size: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 1
    ):
        super().__init__()
        
        self.price_encoder = PriceEncoder(price_input_channels, price_window_size, output_dim=128)
        self.daily_input_channels = int(daily_input_channels)
        self.daily_window_size = int(daily_window_size)
        self.daily_encoder = DailyMacroEncoder(max(1, self.daily_input_channels), max(1, self.daily_window_size), output_dim=32)
        
        # Fusion
        # Input: 128 (price) + 32 (daily) + state_dim + action_dim
        fusion_input_dim = 128 + 32 + state_dim + action_dim
        
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

    def forward(
        self,
        price_seq: torch.Tensor,
        daily_seq: torch.Tensor,
        state_vec: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        price_emb = self.price_encoder(price_seq)
        daily_emb = self.daily_encoder(daily_seq)
        
        x = torch.cat([price_emb, daily_emb, state_vec, action], dim=1)
        q_value = self.net(x)
        
        return q_value
