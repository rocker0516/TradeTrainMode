# Step 流程與 Regime 檢查清單

## 你要求的項目 vs 目前實作

| 項目 | 狀態 | 說明 |
|------|------|------|
| **1. 取得當前 regime（A/C/neutral）與分數（p_up/strength）** | ✅ | • **Action projection** 時：`_apply_regime_action_projection()` 內呼叫 `market_data.get_gate_flags(self.current_step)` 取得 A/C/neutral。<br>• **Reward** 時：step 9 以 `step_idx` 取得 `gate_flags_step`、`regime_score_step` 並傳入 `reward_calculator.compute(..., gate_flags=..., regime_score=...)`。<br>• **Obs**：`_get_observation()` 含 `gate_flags`（one-hot）與 `regime_score`（p_up, p_down, dir_strength）。 |
| **2. 從 policy 拿 action** | ✅ | `step(self, action)` 的 `action` 由外部 policy 傳入。 |
| **3. 連續動作做 action projection（離散則 action mask）** | ✅ | 連續：`_process_action_and_execute()` 內先 cooldown，再 `_apply_regime_action_projection()`：A → clamp [0,1]、C → clamp [-1,0]、neutral → 0。目前無離散 action mask。 |
| **4. 更新倉位、計算成本** | ✅ | `executor.execute(...)` 更新倉位；`cost_calculator.compute(...)` 計算成本並寫入 `info["cost"]`。 |
| **5. 計算 PnL** | ✅ | 主線 reward 為 log-return；`realized_pnl_step`、`unrealized_pnl` 等透過 kwargs 傳入 reward calculator。 |
| **6. 加上 regime 對齊 bonus（reward sign alignment）** | ✅ | • Env 傳入 `gate_flags`、`regime_score` 給 `reward_calculator.compute()`。<br>• `RewardCalculator` / `ConvictionTrendRewardCalculator` 支援 `regime_alignment_bonus_weight`：A 狀態多頭加分、C 狀態空頭加分，依 `dir_strength` 加權；neutral 無 bonus。<br>• 預設 0，可透過 `regime_alignment_bonus_weight`（kwargs 或 `Config.REGIME_ALIGNMENT_BONUS_WEIGHT`）啟用。 |
| **7. 回傳 obs（含 gate one-hot + score）** | ✅ | `return self._get_observation()` 含 `gate_flags`（3 維：A/B/C）、`regime_score`（3 維：p_up, p_down, dir_strength），由 `OBS_STATE_KEYS` 納入。 |

## 程式位置速查

- **Regime 取得（action 用）**：`trading_env.py` → `_apply_regime_action_projection()` → `market_data.get_gate_flags(self.current_step)`。
- **Regime 取得（reward 用）**：`trading_env.py` → `step()` 第 9 段 → `get_gate_flags(step_idx)`、`get_regime_score(step_idx)` 傳入 `reward_calculator.compute(...)`。
- **Action projection**：`trading_env.py` → `_process_action_and_execute()` → `_apply_regime_action_projection(action_used)`。
- **Regime 對齊 bonus**：`Env/Rewards/reward.py` → `RewardCalculator.compute()` / `ConvictionTrendRewardCalculator.compute()`，使用 `gate_flags`、`regime_score`、`regime_alignment_bonus_weight`。
- **Obs gate + score**：`Env/Components/observer.py` → `_get_gate_obs()`、`_get_regime_score_obs()`；`Config.OBS_STATE_KEYS` 含 `gate_flags`、`regime_score`。

## 啟用 Regime 對齊 Bonus

- 建 env 時：`TradingEnvironment(..., regime_alignment_bonus_weight=0.1)`
- 或於 `Env/config.py` 設定 `REGIME_ALIGNMENT_BONUS_WEIGHT = 0.1`（預設 0）。
