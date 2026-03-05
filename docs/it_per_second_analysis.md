# it/s 僅約 400 多的原因分析與優化建議

## 1. 當前指標解讀

- **it/s ≈ 452**：表示每秒約 **452 個 timesteps**（即 452 次環境步）。
- 使用 **64 個並行環境**（SubprocVecEnv），因此每秒約執行 **452/64 ≈ 7 輪**「collect 64 步 + 可能 train」。
- 每輪約 **140ms**，其中可能包含：64 個 env.step()、主進程 predict、callback、以及每 2 步一次的 SAC 梯度更新。

---

## 2. 瓶頸分類與可能原因

### 2.1 環境步（env.step）— 最可能的主因

- **SubprocVecEnv**：每輪主進程發送 64 個 action，64 個子進程各自執行 `env.step(action)`，再將 `(obs, reward, done, info)` 序列化回傳主進程。
- **單步成本高**：
  - 每步回傳的 obs 對應「下一步」狀態（`current_step` 已 +1），故需在回傳前算一次 `get_market_metrics` 與 `compute_risk_signals`；**reset() 已優化**為傳入已算好的 metrics/risk，避免在 `_get_observation()` 內重算。
  - 每步還有：`get_price_seq` / `get_1d_seq`（可能 `np.vstack` 複製）、`_process_action_and_execute`、reward/cost、`_build_step_info`、`tracker.update_account_series` 等。
- **Observation 體積**：obs 為 Dict，含多個 Box（如 `price_seq_target` 36×F1、`price_seq_others` 36×F2、1d 序列、account_state、context_state）。單 env 約數千 float16/float32，64 個 env 每輪需序列化/反序列化與跨進程傳輸，加重 **IPC 與 pickle 開銷**。
- **結論**：若瓶頸在 **env.step**，會表現為：CPU 被 64 個子進程與主進程的收發/反序列化佔滿，it/s 受「最慢的 env + IPC」限制。

### 2.2 主進程：policy predict + SAC 訓練

- **predict**：每輪將 64 筆 Dict obs 轉成 tensor（若用 GPU 還有 CPU→GPU），再跑 policy forward；通常比 64 個 env.step 快。
- **train**：`TRAIN_FREQ=2`、`GRADIENT_STEPS=2` 表示每 2 個 timestep 做 2 次梯度更新（每次 batch=256）。相當於 **每步約 1 次梯度更新**，backward + 優化器會佔用 GPU/CPU 時間。
- 若 **GPU 或 replay buffer 取樣**成為瓶頸，it/s 會對「減少 train 頻率」或「增大 n_envs」有反應。

### 2.3 Callback 開銷

- **每步**：`_on_step` 遍歷 64 筆 `infos`、往 deque 追加 cost、檢查 episode 結束；開銷相對小。
- **每 1000 步**：更新 Lagrangian λ。
- **每 2000 步**（`_sync_lambda_every_n_updates=2`）：`training_env.env_method("set_lagrangian_lambdas", ...)`，對 **64 個子進程各做一次方法呼叫**，等於 64 次 IPC，可能造成週期性小尖峰。
- **每 log_freq 個 episode**：`_dump_stats` 會做 replay buffer 採樣與 critic forward（`_get_q_values_stats`），額外 GPU/CPU 負載，但非每步。

### 2.4 其他設定

- **OMP_NUM_THREADS=1、MKL_NUM_THREADS=1**：已設，有助避免與 64 子進程搶 CPU，是正確方向。
- **torch.set_num_threads**：主進程已依 n_envs 限制線程數，合理。
- **VecMonitor**：每步寫入監控資料，若寫檔或統計過重也可能有一點影響，一般小於 env.step。

---

## 3. 如何確認瓶頸：Profile 模式

專案已支援 **profile 模式**，可量測「env.step」與「model.predict」的相對時間，直接判斷瓶頸在主進程還是環境：

```bash
& .conda/python.exe Train/run_sac_lag.py --profile_steps 5000
```

輸出會類似：

- `[Profile] env.step(): X s (Y%); model.predict(): Z s (W%)`
- 若 **env.step 佔比明顯較高** → 瓶頸在環境與 SubprocVecEnv IPC，應優化 env 單步成本與 obs 體積、或調整 n_envs。
- 若 **model.predict 佔比明顯較高** → 瓶頸在 policy/GPU，可考慮 `--device cuda`、`--train_freq 4` 等。

**注意**：profile 未含「SAC 的 train()」與 callback 的完整開銷，僅對比 step vs predict；但已足夠判斷是 env 重還是主進程推理重。

---

## 4. 優化建議（按優先級）

### 4.1 先跑 Profile（必做）

- 執行上述 `--profile_steps 5000`，根據 env.step % 與 predict % 決定下面要主攻「環境」還是「主進程」。

### 4.2 若瓶頸在環境 / IPC

1. **減少重複計算**  
   - **reset()** 已改為傳入 `precomputed_metrics` / `precomputed_risk_signals` 給 `_get_observation()`，避免重算。`step()` 回傳的 obs 對應下一步，無法重用本步開頭的 metrics，故無重複可削。
2. **降低 obs 體積**  
   - 維持 `OBS_DTYPE="float16"`（若尚未使用可改）；或縮小 `window_size` / 特徵維度（在不大幅影響策略前提下），以減輕 64 份 obs 的序列化與 IPC。
3. **試減 n_envs**  
   - 例如改為 `--n_envs 32`。若 it/s 反而上升或持平，代表 64 子進程 + IPC 已過載；若 it/s 下降，則可再試 48 或保持 64，並搭配 1、2。
4. **Lambda 同步頻率**  
   - 在 `LagrangianCallback` 中將 `_sync_lambda_every_n_updates` 調大（例如 4），減少 `env_method(..., set_lagrangian_lambdas)` 的次數，降低 IPC 尖峰。

### 4.3 若瓶頸在主進程 / GPU

1. **降低訓練頻率**  
   - `--train_freq 4 --gradient_steps 4`：每 4 步做 4 次梯度更新，總更新量相近，但每步的 train 次數減半，有助提升 it/s。
2. **確認使用 GPU**  
   - `--device cuda`（你目前應已是），避免 policy 與 train 全在 CPU。
3. **拉高 n_envs（在 CPU/記憶體允許下）**  
   - 若單步 train 成本高，略增 n_envs 可讓「每步收集的資料量」變大，攤薄每 step 的 train 相對成本（需與 4.2 的 IPC 取捨）。

### 4.4 通用

- **CHECKPOINT_SAVE_FREQ**：保持較大（如 1_000_000），避免頻繁寫盤。
- **LOG_EVERY_EPISODES / log_freq**：若不需要太細的日誌，可適度調大，減少 `_dump_stats` 與 Q 值統計的觸發次數。

---

## 5. 小結

| 可能原因 | 說明 | 建議動作 |
|----------|------|----------|
| 64 個 env.step 單步過重 | 重複的 get_market_metrics / compute_risk_signals、序列體積大 | 合併重複計算、縮小 obs、必要時減 n_envs |
| SubprocVecEnv IPC | 64 份 obs + info 每輪序列化/傳輸 | 減小 obs、降低 lambda sync 頻率、試 n_envs=32 |
| SAC 每步約 1 次梯度更新 | TRAIN_FREQ=2, GRADIENT_STEPS=2 | 試 train_freq=4, gradient_steps=4 |
| Lambda 同步 64 次 IPC | 每 2000 步 env_method 一次 | 增大 _sync_lambda_every_n_updates |

**第一步**：執行 `--profile_steps 5000` 看 env.step vs predict 佔比，再依上述表格與 4.2 / 4.3 做對應優化。
