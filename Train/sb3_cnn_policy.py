from __future__ import annotations

"""
SB3（stable-baselines3）用的多輸入觀測編碼器：雙分支 CNN + 向量 MLP

為什麼要這樣做？
- 你的 observation 是 Dict：
  - price_seq: (window_size, F_5m)     # 5m 序列（主幹；F_5m 會隨特徵集合擴充）
  - price_seq_1d: (window_size_1d, F_1d)  # 1d 序列（regime 背景）
  - account_state/time_state/rhythm_state/cost_state: 向量特徵
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
- cost_state 仍會被 MLP 分支吃進去，讓策略能看到成本/風險接近程度。
"""

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


class DualCnnFeatureExtractor(BaseFeaturesExtractor):
    """
    雙分支 CNN 特徵抽取器（SB3 MultiInputPolicy）。

    結構：
    - 5m 分支（較深）：Conv1d -> Conv1d -> Conv1d -> GAP -> Linear -> emb_5m
    - 1d 分支（小 CNN）：Conv1d -> Conv1d -> GAP -> Linear -> emb_1d
    - 向量分支（MLP）：concat(account/time/rhythm/cost) -> MLP -> emb_vec
    - 融合：concat(emb_5m, emb_1d, emb_vec) -> Linear -> out_dim
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        emb_5m: int = 128,
        emb_1d: int = 64,
        emb_vec: int = 128,
        out_dim: int = 256,
    ) -> None:
        # BaseFeaturesExtractor 需要指定 features_dim（也就是本 extractor 的輸出維度）
        super().__init__(observation_space, features_dim=int(out_dim))

        # ---- 解析 observation_space ----
        # price_seq: (window, F_5m)
        seq_5m_shape = observation_space.spaces["price_seq"].shape
        seq_1d_shape = observation_space.spaces["price_seq_1d"].shape
        if seq_5m_shape is None or seq_1d_shape is None:
            raise ValueError("price_seq / price_seq_1d 必須是具體 shape 的 spaces.Box")

        win_5m, feat_5m = int(seq_5m_shape[0]), int(seq_5m_shape[1])
        win_1d, feat_1d = int(seq_1d_shape[0]), int(seq_1d_shape[1])

        # 向量分支輸入維度（把所有向量 state 串起來）
        vdim = 0
        for k in ("account_state", "time_state", "rhythm_state", "cost_state"):
            shp = observation_space.spaces[k].shape
            if shp is None or len(shp) != 1:
                raise ValueError(f"{k} 必須是 1D 向量 spaces.Box")
            vdim += int(shp[0])

        # ---- 5m CNN（主幹：對 288 長序列更有力）----
        # kernel 取 7/5/3 是常見穩定配置：先抓較粗型態，再細化
        self.cnn_5m = nn.Sequential(
            nn.Conv1d(in_channels=feat_5m, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),  # (B,64,1)
            nn.Flatten(),  # (B,64)
            nn.Linear(64, emb_5m),
            nn.ReLU(),
        )

        # ---- 1d 小 CNN（短序列：30）----
        self.cnn_1d = nn.Sequential(
            nn.Conv1d(in_channels=feat_1d, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),  # (B,32,1)
            nn.Flatten(),  # (B,32)
            nn.Linear(32, emb_1d),
            nn.ReLU(),
        )

        # ---- 向量 MLP（成本/帳戶/時間等結構化特徵）----
        self.mlp_vec = nn.Sequential(
            nn.Linear(vdim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_vec),
            nn.ReLU(),
        )

        # ---- 融合層 ----
        fusion_in = int(emb_5m + emb_1d + emb_vec)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, int(out_dim)),
            nn.ReLU(),
        )

        # 保存一些資訊，方便 debug
        self._meta: Dict[str, Tuple[int, int]] = {
            "price_seq": (win_5m, feat_5m),
            "price_seq_1d": (win_1d, feat_1d),
            "vec_dim": (vdim, 1),
        }

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # SB3 可能把 env 的 float16 observation 原樣送進來（以省 replay buffer RAM）。
        # 為了讓卷積/MLP 訓練更穩定，這裡統一轉回 float32 做計算。
        obs = {k: v.float() for k, v in observations.items()}

        # ---- 5m 分支 ----
        x5 = _as_bcl(obs["price_seq"])
        e5 = self.cnn_5m(x5)

        # ---- 1d 分支 ----
        x1 = _as_bcl(obs["price_seq_1d"])
        e1 = self.cnn_1d(x1)

        # ---- 向量分支 ----
        v = torch.cat(
            [
                obs["account_state"],
                obs["time_state"],
                obs["rhythm_state"],
                obs["cost_state"],
            ],
            dim=1,
        )
        ev = self.mlp_vec(v)

        # ---- 融合 ----
        fused = torch.cat([e5, e1, ev], dim=1)
        return self.fusion(fused)

