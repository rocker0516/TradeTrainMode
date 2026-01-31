# Lambda（Lagrangian 乘數）調教優化

## 一、現況公式

目前每條成本線使用 **P-Control** 更新 λ：

```
λ_new = clamp(λ_old + kp * (avg_cost - cost_limit), lambda_min, lambda_max)
```

- **avg_cost**：最近 `update_freq * n_envs` 步的 step-level cost 平均（例如 update_freq=1、n_envs=64 → 64 筆）。
- **kp**：目前全通道固定 0.1。
- **lambda_init**：0、**lambda_max**：5。

因此 λ 會隨「違規程度」(avg_cost - limit) 線性增減，並被 clamp 在 [0, 5]。

---

## 二、常見問題

1. **λ 震盪**：kp 過大或 avg 視窗太短 → 違規一高就猛升 λ、一低又猛降，難以收斂。
2. **λ 反應太慢**：kp 過小或視窗過長 → 違規持續很久 λ 才上來，約束生效太晚。
3. **尺度不一致**：risk（0/1）、fric（約 0.0001）、sl_buf（0~1）同一 kp，敏感度差異大。
4. **早期崩潰**：lambda_init=0 時，前幾千步幾乎沒懲罰，policy 可能先學到高風險行為再被拉回。

---

## 三、優化方向

### 3.1 參數可調（優先）

- **per-channel kp**：risk 可略大（例如 0.2）、fric 可略小（例如 0.05），依「希望多快反應」與「cost 尺度」調整。
- **lambda_init**：若希望一開始就壓制風險，可設 risk 的 lambda_init=0.01～0.1，其餘維持 0。
- **lambda_max**：risk 可設大一點（例如 10）讓死亡成本懲罰夠重；fric 可維持 5 或更小。

**實作**：在 `TrainConfig` / `run_sac_lag` 用 `--kp`、`--lambda_init`、`--lambda_max`（或 per-channel 覆寫）傳入 `LagrangianChannelConfig`。

### 3.2 拉長 cost 平均視窗

- **現狀**：buffer 長度 = `update_freq * n_envs`（例如 1×64=64），avg_cost 易受少數步影響而抖動。
- **做法**：另設 `cost_window_steps`（例如 500～2000），讓 buffer 長度 = `cost_window_steps * n_envs`，仍每 `update_freq` 步更新一次 λ，但用更長區間的平均違規來更新，λ 較平滑、較少震盪。

**實作**：`LagrangianCallback` 增加參數 `cost_window_steps`，預設等於 `update_freq`（維持舊行為），設大即拉長視窗。

### 3.3 Lambda warmup（可選）

- **做法**：前 N 步（例如 5k～20k）對 λ 更新打折扣（例如 effective_kp = kp * min(1, step / warmup_steps)），或前 N 步不更新 λ，讓 policy 先稍微探索再加重約束。
- **效果**：減少「一開局就因 λ 飆高而幾乎只優化 cost、reward 學不到」的狀況。

**實作**：在 Controller 或 Callback 內依 `num_timesteps` 對 kp 或 violation 做縮放。

### 3.4 PI / PID 控制（進階）

- **P-only**：目前做法，易震盪。
- **PI**：在 λ 更新中加上「違規的累積」（積分項），例如 `λ_new = λ_old + kp * violation + ki * integral(violation)`，再 clamp。可減少穩態誤差，但需調 ki 與 integral 衰減。
- **PID**：再加 derivative（違規變化率），反應更快、但對雜訊敏感，實作與調參更複雜。

**實作**：需在 `SharedLagrangianController` / `MultiSharedLagrangianController` 維護每通道的 integral（與可選 derivative），並在 `update()` 使用 ki、kd。可先做 PI、再視需要加 D。

### 3.5 自適應 kp（可選）

- **做法**：違規大時用較大 kp、違規小時用較小 kp，例如 `effective_kp = kp_base * clip(|violation| / threshold, 0.5, 2.0)`，或依 violation 正負與大小分段。
- **效果**：大違規時快速拉高 λ、接近 limit 時放慢，減少 overshoot。

**實作**：在 `update()` 內依 `violation` 計算 `effective_kp` 再代入現有公式。

### 3.6 Cost limit 排程（可選）

- **做法**：訓練前期用較鬆的 cost_limit（例如 1.2× 目標），隨步數線性或階段性縮緊到目標值。
- **效果**：類似 curriculum，先讓 policy 在較寬鬆約束下學到合理行為，再收緊。

**實作**：在 Callback 或 Controller 依 `num_timesteps` 計算當前 cost_limit，再傳入 update。

---

## 四、建議優先順序

| 優先級 | 項目 | 預期效果 | 改動範圍 |
|--------|------|----------|----------|
| 1 | 參數可調：kp / lambda_init / lambda_max（含 per-channel） | 依風險/手續費/止損分開調敏感度與上下界 | TrainConfig、run_sac_lag、LagrangianChannelConfig |
| 2 | 拉長 cost 視窗（cost_window_steps） | λ 更新更平滑、減少震盪 | LagrangianCallback |
| 3 | Lambda warmup | 前期少震盪、約束漸強 | Controller 或 Callback |
| 4 | PI 項（integral of violation） | 減少穩態違規、更穩收斂 | Controller.update() |
| 5 | 自適應 kp / cost limit 排程 | 進階穩定與課程學習 | Controller / Callback |

建議先做 **1 + 2**，觀察 TensorBoard 的 `lagrangian/lambda_*`、`lagrangian/cost_violation_*` 是否較平滑、約束是否在合理步數內收斂；若仍震盪再考慮 3～5。

---

## 五、監控建議

- **TensorBoard**：  
  - `lagrangian/lambda_risk`、`lambda_fric`、`lambda_sl_buf`、`lambda_sl_event`  
  - `lagrangian/avg_cost_*`、`lagrangian/cost_violation_*`  
- **解讀**：  
  - λ 長期單向上升且 violation 仍正 → 可能 limit 過嚴或 kp 過大，可略鬆 limit 或略降 kp。  
  - λ 與 violation 來回震盪 → 拉長 cost 視窗或略降 kp、或加 warmup/PI。  
  - λ 長期接近 0 且 violation 常負 → 約束易滿足，可視需求略嚴 limit 或略增 kp 讓約束更明確。
