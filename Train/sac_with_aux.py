"""
SAC with Auxiliary Loss：預測「下一步報酬符號」（做法二方案 C）

僅在有持倉的 transition 上計算 auxiliary loss，讓網路顯式學「持倉方向 ↔ 盈虧方向」。
需搭配 OptimizedDictReplayBuffer（回傳 DictReplayBufferSamplesWithAux）與
DualCnnFeatureExtractor.forward_with_aux() 使用。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.sac import SAC
from Train.optimized_dict_replay_buffer import DictReplayBufferSamplesWithAux


# 報酬符號對應類別：負 -> 0，近零 -> 1，正 -> 2（與 equity 報酬方向一致：漲=賺、跌=虧）
_THRESH = 1e-6

# 供 TensorBoard / 除錯：類別對應關係
AUX_CLASS_NAMES = ("down", "flat", "up")  # 0, 1, 2


def _step_log_return_to_class(step_log_returns: th.Tensor) -> th.Tensor:
    """將 step_log_return 轉成 3 類 (0=跌, 1=平, 2=漲)，long 型供 cross_entropy。"""
    r = step_log_returns.view(-1)
    out = th.ones(r.shape[0], dtype=th.long, device=r.device)
    out[r < -_THRESH] = 0
    out[r > _THRESH] = 2
    return out


class SACWithAuxiliaryLoss(SAC):
    """
    在 SAC 的 actor 更新中加入 auxiliary loss：預測「下一步報酬符號」。
    僅對 replay_data 中有持倉的樣本計算 CE loss，使模型學到持倉方向與盈虧方向的對應。
    """

    def __init__(
        self,
        *args,
        aux_coef: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.aux_coef = float(aux_coef)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        aux_losses_list: list[float] = []
        aux_n_pos_list: list[float] = []  # 診斷：每個 batch 有持倉樣本數
        aux_has_data_count = 0  # 診斷：有 aux 資料的 step 數
        aux_acc_list: list[float] = []  # 方向驗證：預測類別 vs 目標類別準確率
        aux_target_frac_sum: list[float] = [0.0, 0.0, 0.0]  # 目標類別 0/1/2 累計比例
        aux_target_count = 0

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Auxiliary loss：預測「下一步報酬符號」；有持倉時僅在有持倉樣本上算 loss，否則用全 batch 避免恆為 0
            step_log_returns = getattr(replay_data, "step_log_returns", None)
            has_positions = getattr(replay_data, "has_positions", None)
            if (
                step_log_returns is not None
                and has_positions is not None
                and hasattr(self.policy.features_extractor, "forward_with_aux")
            ):
                aux_has_data_count += 1
                _, aux_logits = self.policy.features_extractor.forward_with_aux(replay_data.observations)
                target_class = _step_log_return_to_class(step_log_returns)
                mask = (has_positions.view(-1) > 0.5).float()
                n_pos = mask.sum().item()
                aux_n_pos_list.append(float(n_pos))

                aux_loss_per_sample = F.cross_entropy(aux_logits, target_class, reduction="none")
                if n_pos > 0:
                    aux_loss = (aux_loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
                else:
                    # 無持倉樣本時用全 batch 平均，讓 aux 頭有梯度且 log 不恆為 0
                    aux_loss = aux_loss_per_sample.mean()
                actor_loss = actor_loss + self.aux_coef * aux_loss
                aux_losses_list.append(aux_loss.item())

                # 方向驗證：預測 vs 目標準確率、目標類別分佈（確認 0=跌/1=平/2=漲 對齊）
                with th.no_grad():
                    pred_class = aux_logits.argmax(dim=-1)
                    acc = (pred_class == target_class).float().mean().item()
                    aux_acc_list.append(acc)
                    n = target_class.shape[0]
                    for k in range(3):
                        aux_target_frac_sum[k] += (target_class == k).float().sum().item()
                    aux_target_count += n

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # 一律寫入；有 aux 資料時必有一筆 loss（有持倉用 mask，無持倉用全 batch）
        self.logger.record(
            "train/aux_return_sign_loss",
            np.mean(aux_losses_list) if len(aux_losses_list) > 0 else 0.0,
        )
        # 一律寫入，讓 TensorBoard 一定看得到（無 aux 資料時為 0）
        self.logger.record(
            "train/aux_batch_n_pos",
            np.mean(aux_n_pos_list) if len(aux_n_pos_list) > 0 else 0.0,
        )
        self.logger.record(
            "train/aux_has_data_ratio",
            float(aux_has_data_count) / float(gradient_steps) if gradient_steps > 0 else 0.0,
        )
        # 方向驗證：準確率 > 1/3 表示有學到方向；目標分佈可確認標籤 0=跌/1=平/2=漲 是否合理
        if len(aux_acc_list) > 0:
            self.logger.record("train/aux_accuracy", np.mean(aux_acc_list))
        if aux_target_count > 0:
            total = float(aux_target_count)
            for k, name in enumerate(AUX_CLASS_NAMES):
                self.logger.record(f"train/aux_target_frac_{name}", aux_target_frac_sum[k] / total)
