from __future__ import annotations

"""
SB3（stable-baselines3）用的多輸入觀測編碼器：雙分支 CNN + 向量 MLP

為什麼要這樣做？
- 你的 observation 是 Dict：
  - price_seq: (window_size, F_5m)     # 5m 序列（主幹；F_5m 會隨特徵集合擴充）
  - price_seq_1d: (window_size_1d, F_1d)  # 1d 序列（regime 背景）
  - account_state/cost_state: 向量特徵
- 我們希望「5m / 1d 使用不同 CNN 設計」：
  - 5m 序列長（288），用稍深一點的 CNN 抽型態
  - 1d 序列短（30），用小 CNN 即可，避免過度容量/過擬合

使用方式（訓練端）：
    from stable_baselines3 import SAC
    from Train.sb3_cnn_policy import DualCnnFeatureExtractor

    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=DualCnnFeatureExtractor,
            features_extractor_kwargs=dict(out_dim=256),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        ),
        device="cuda",
    )

注意：
- 本檔只負責「特徵抽取器」，不負責 cost/Lagrangian 更新。
- cost_state 為可選字段：如果 Env 端不提供，會使用零向量填充（Env 端不應包含成本信息，成本應在 Train 端通過 wrapper 處理）。
"""

import math
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _as_bcl(x: torch.Tensor) -> torch.Tensor:
    """
    將輸入轉成 Conv1d 需要的 shape: (B, C, L)

    你的序列觀測是 (B, L, F)，其中：
    - L: time window
    - F: feature channels

    Conv1d 的 C 應該對應 feature channels，因此需要 transpose。
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (B,L,F), got shape={tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


class Target5mCNN(nn.Module):
    """
    Target Symbol 5m CNN（精简特征 + 关键缺失特征）：
    - 输入：(B, L=432, C=28-30) - target_symbol 的精简特征（移除长时尺度干扰，添加关键价位和订单簿）
    - 架构：3层Conv1d（保持原有深度）
    - 输出：emb_target_5m (128维)
    """
    def __init__(self, in_channels: int = 30, emb_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2).contiguous()
        return self.cnn(x)


class Others5mCNN(nn.Module):
    """
    Other Symbols 5m CNN（紧凑特征）：
    - 输入：(B, L=432, C=32-40) - 4个币种 * 8-10通道
    - 架构：2层Conv1d（较浅，因为特征已压缩）
    - 输出：emb_others_5m (64维)
    """
    def __init__(self, in_channels: int = 40, emb_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return self.cnn(x)


class Target1dCNN(nn.Module):
    """
    Target Symbol 1d CNN（详细特征 + 长期关键价位 + 时间特征）：
    - 输入：(B, L=30, C=18) - target_symbol 的详细 1d 特征（包含长期关键价位、VWAP、时间特征）
    - 输出：emb_target_1d (64维)
    """
    def __init__(self, in_channels: int = 18, emb_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return self.cnn(x)


class Others1dCNN(nn.Module):
    """
    Other Symbols + Macro 1d CNN（紧凑特征）：
    - 输入：(B, L=30, C=28) - 4个币种 * 6通道 + 4个macro
    - 输出：emb_others_1d (32维)
    """
    def __init__(self, in_channels: int = 28, emb_dim: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, emb_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return self.cnn(x)


class CrossTimeframeAttention(nn.Module):
    """
    5m 和 1d 特征之间的交互：
    - Target 5m <-> Target 1d（主要交互）
    - Others 5m <-> Others 1d（辅助交互）
    """
    def __init__(self, emb_5m: int = 192, emb_1d: int = 96, hidden_dim: int = 128):
        super().__init__()
        # emb_5m = 128(target) + 64(others)
        # emb_1d = 64(target) + 32(others)
        self.q_proj = nn.Linear(emb_5m, hidden_dim)
        self.k_proj = nn.Linear(emb_1d, hidden_dim)
        self.v_proj = nn.Linear(emb_1d, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_5m)
        self.hidden_dim = hidden_dim
        
    def forward(self, emb_5m: torch.Tensor, emb_1d: torch.Tensor) -> torch.Tensor:
        # 5m 关注 1d 的宏观背景
        # emb_5m: (B, 192), emb_1d: (B, 96)
        q = self.q_proj(emb_5m)  # (B, hidden_dim)
        k = self.k_proj(emb_1d)  # (B, hidden_dim)
        v = self.v_proj(emb_1d)  # (B, hidden_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # (B, hidden_dim)
        attn = torch.softmax(attn_scores, dim=-1)  # (B, hidden_dim)
        out = torch.matmul(attn, v)  # (B, hidden_dim)
        out = self.out_proj(out)  # (B, emb_5m)
        return out + emb_5m  # 残差连接


class DualCnnFeatureExtractor(BaseFeaturesExtractor):
    """
    新架构：双CNN分离 + Cross-Attention
    1. Target 5m CNN -> emb_target_5m (128)
    2. Others 5m CNN -> emb_others_5m (64)
    3. Target 1d CNN -> emb_target_1d (64)
    4. Others 1d CNN -> emb_others_1d (32)
    5. 融合 5m: concat(emb_target_5m, emb_others_5m) -> emb_5m (192)
    6. 融合 1d: concat(emb_target_1d, emb_others_1d) -> emb_1d (96)
    7. Cross-Attention（5m <-> 1d）
    8. 最终融合 + 向量分支
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        emb_5m_target: int = 128,
        emb_5m_others: int = 64,
        emb_1d_target: int = 64,
        emb_1d_others: int = 32,
        emb_vec: int = 128,
        out_dim: int = 256,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim=int(out_dim))

        # ---- 解析 observation_space ----
        # 新的分离 observation space
        seq_5m_target_shape = observation_space.spaces["price_seq_target"].shape
        seq_5m_others_shape = observation_space.spaces["price_seq_others"].shape
        seq_1d_target_shape = observation_space.spaces["price_seq_1d_target"].shape
        seq_1d_others_shape = observation_space.spaces["price_seq_1d_others"].shape
        
        if (seq_5m_target_shape is None or seq_5m_others_shape is None or 
            seq_1d_target_shape is None or seq_1d_others_shape is None):
            raise ValueError("所有 price_seq 相关空间必须具有具体 shape")

        win_5m = int(seq_5m_target_shape[0])
        feat_5m_target = int(seq_5m_target_shape[1])
        feat_5m_others = int(seq_5m_others_shape[1])
        win_1d = int(seq_1d_target_shape[0])
        feat_1d_target = int(seq_1d_target_shape[1])
        feat_1d_others = int(seq_1d_others_shape[1])

        # 向量分支輸入維度
        # account_state 必須存在；context_state、cost_state 為可選
        if "account_state" not in observation_space.spaces:
            raise ValueError("observation_space 必須包含 'account_state'")
        account_shp = observation_space.spaces["account_state"].shape
        if account_shp is None or len(account_shp) != 1:
            raise ValueError("account_state 必須是 1D 向量 spaces.Box")
        vdim = int(account_shp[0])

        # context_state：上一動執行結果（action 是否被覆寫、target/final、預測強平/餘額等）
        self.has_context_state = "context_state" in observation_space.spaces
        if self.has_context_state:
            ctx_shp = observation_space.spaces["context_state"].shape
            if ctx_shp is None or len(ctx_shp) != 1:
                raise ValueError("context_state 必須是 1D 向量 spaces.Box")
            self.context_state_dim = int(ctx_shp[0])
            vdim += self.context_state_dim
        else:
            self.context_state_dim = 0

        # cost_state 為可選（如果不存在，使用零向量）
        self.has_cost_state = "cost_state" in observation_space.spaces
        if self.has_cost_state:
            cost_shp = observation_space.spaces["cost_state"].shape
            if cost_shp is None or len(cost_shp) != 1:
                raise ValueError("cost_state 必須是 1D 向量 spaces.Box")
            self.cost_state_dim = int(cost_shp[0])
            vdim += self.cost_state_dim
        else:
            self.cost_state_dim = 0

        # ---- 双CNN架构 ----
        self.cnn_5m_target = Target5mCNN(in_channels=feat_5m_target, emb_dim=emb_5m_target)
        self.cnn_5m_others = Others5mCNN(in_channels=feat_5m_others, emb_dim=emb_5m_others)
        self.cnn_1d_target = Target1dCNN(in_channels=feat_1d_target, emb_dim=emb_1d_target)
        self.cnn_1d_others = Others1dCNN(in_channels=feat_1d_others, emb_dim=emb_1d_others)

        # ---- Cross-Attention ----
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            emb_5m_fused = emb_5m_target + emb_5m_others
            emb_1d_fused = emb_1d_target + emb_1d_others
            self.cross_attn = CrossTimeframeAttention(
                emb_5m=emb_5m_fused,
                emb_1d=emb_1d_fused,
                hidden_dim=128
            )

        # ---- 向量 MLP ----
        self.mlp_vec = nn.Sequential(
            nn.Linear(vdim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_vec),
            nn.ReLU(),
        )

        # ---- 融合層 ----
        # 如果使用 cross-attention，5m 特征已经被增强
        fusion_in = int(emb_5m_target + emb_5m_others + emb_1d_target + emb_1d_others + emb_vec)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, int(out_dim)),
            nn.ReLU(),
        )

        # 保存一些資訊，方便 debug
        self._meta: Dict[str, Tuple[int, int]] = {
            "price_seq_target": (win_5m, feat_5m_target),
            "price_seq_others": (win_5m, feat_5m_others),
            "price_seq_1d_target": (win_1d, feat_1d_target),
            "price_seq_1d_others": (win_1d, feat_1d_others),
            "vec_dim": (vdim, 1),
        }

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # SB3 可能把 env 的 float16 observation 原樣送進來（以省 replay buffer RAM）。
        # 僅在非 float32 時轉換，避免冗餘的 dict 與 tensor 複製（train 時若已是 float32 可省）。
        if all(v.dtype == torch.float32 for v in observations.values()):
            obs = observations
        else:
            obs = {k: v.float() for k, v in observations.items()}

        # ---- 5m 分支（分离）----
        e5_target = self.cnn_5m_target(obs["price_seq_target"])
        e5_others = self.cnn_5m_others(obs["price_seq_others"])
        e5_fused = torch.cat([e5_target, e5_others], dim=1)  # (B, 192)

        # ---- 1d 分支（分离）----
        e1_target = self.cnn_1d_target(obs["price_seq_1d_target"])
        e1_others = self.cnn_1d_others(obs["price_seq_1d_others"])
        e1_fused = torch.cat([e1_target, e1_others], dim=1)  # (B, 96)

        # ---- Cross-Attention ----
        if self.use_cross_attention:
            e5_fused = self.cross_attn(e5_fused, e1_fused)  # (B, 192)

        # ---- 向量分支 ----
        # account_state 必須存在；context_state、cost_state 為可選
        vec_list = [obs["account_state"]]
        if self.has_context_state:
            vec_list.append(obs["context_state"])
        else:
            batch_size = obs["account_state"].shape[0]
            device = obs["account_state"].device
            vec_list.append(torch.zeros(batch_size, self.context_state_dim, device=device, dtype=obs["account_state"].dtype))
        if self.has_cost_state:
            vec_list.append(obs["cost_state"])
        else:
            batch_size = obs["account_state"].shape[0]
            device = obs["account_state"].device
            vec_list.append(torch.zeros(batch_size, self.cost_state_dim, device=device, dtype=obs["account_state"].dtype))

        v = torch.cat(vec_list, dim=1)
        ev = self.mlp_vec(v)

        # ---- 融合 ----
        fused = torch.cat([e5_fused, e1_fused, ev], dim=1)
        return self.fusion(fused)

