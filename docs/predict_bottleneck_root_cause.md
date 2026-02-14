# Predict 瓶頸根因分析（主進程/GPU）

## 1. 結論：沒有「多算一次」的冗餘，瓶頸在單次 forward 的必經路徑

- 已確認：**沒有**同一筆 obs 被 predict 或 forward 算兩次。
- 瓶頸在「每一 step 必經的單次路徑」：**obs（numpy float16）→ 轉 tensor → .float() → 4×CNN + Cross-Attention + MLP + fusion**。
- 以下優化都是「減少這條路徑上的開銷」，而不是「砍掉重複計算」。

---

## 2. 資料流與每步必經路徑

### 2.1 每 step 主進程在做什麼（SB3 off-policy collect）

1. **predict(obs)**  
   - 輸入：`obs` = Dict[str, np.ndarray]，來自上一步的 `next_obs`（或 reset），shape 為 `(n_envs, ...)`，dtype 為 **float16**（Env `OBS_DTYPE`）。
   - SB3 內部：numpy → tensor（並放到 device），再呼叫 policy。
   - 我們的 `DualCnnFeatureExtractor.forward`：對整個 Dict 做 `obs = {k: v.float() for k, v in observations.items()}`，再 4×CNN、Cross-Attention、MLP、fusion。
   - 沒有在其他地方再對同一份 obs 做一次 predict/forward。

2. **env.step(actions)**  
   - 64 個子進程並行，回傳 next_obs（float16）、rewards、dones、infos。

3. **buffer.add(obs, next_obs, ...)**  
   - 把當前 obs、next_obs 寫進 replay buffer（float16 存盤）。
   - 內部用 `np.array(obs[key])` 寫入，會多一次完整複製。

4. **train（每 train_freq 步）**  
   - 從 buffer sample 出 batch（numpy float16）→ `to_torch` → 進 policy/critic。
   - 同樣會進 `DualCnnFeatureExtractor.forward`，再做一次 `.float()` + 4×CNN + Cross-Attention + MLP。

所以：**predict 與 train 各自只有「一條」forward，沒有同一筆 obs 被重複 forward。**

---

## 3. 瓶頸拆解（單次 predict 的必經路徑）

| 階段 | 說明 | 可否省略 |
|------|------|----------|
| numpy → tensor | SB3 把 VecEnv 的 Dict numpy 轉成 tensor、搬到 device | 否（框架必經） |
| Dict 每 key `.float()` | 我們在 forward 裡把 float16 轉 float32 | 否（數值穩定需要）；可改為「僅在非 float32 時轉」 |
| 4×CNN forward | price_seq_target/others、price_seq_1d_target/others | 否（主幹計算） |
| Cross-Attention | 5m ↔ 1d | 可選（關掉會變快，但會改表現） |
| MLP(vec) + fusion | 向量特徵與融合 | 否 |

也就是說：**沒有「多算一次」的 bug，只有「這條路徑本身就很重」**。能做的只有：
- 減少這條路徑上的冗餘（例如 .float 條件化、buffer 寫入方式）、以及
- 加速 forward 本身（例如 torch.compile）。

---

## 4. 已做／建議的程式層優化（非調參）

### 4.1 已實作

1. **DualCnnFeatureExtractor.forward：僅在非 float32 時做 .float()**  
   - 若 SB3 / buffer 在某些路徑已給 float32，就不要再建新 Dict 且不重複 .float()，可省一點記憶體與算力。
   - 多數情況 env 與 buffer 仍是 float16，所以 predict 時仍會做一次 .float()，但 train 時若未來有 float32 路徑即可受益。

2. **OptimizedDictReplayBuffer.add：用 np.copyto 寫入 obs/next_obs**  
   - 不再用 `self.observations[key][self.pos] = np.array(obs[key])`（會多一個臨時陣列）。
   - 改為 `np.copyto(self.observations[key][self.pos], obs[key])`，直接寫入預先配置的 slot，避免多一次大陣列分配與複製。

### 4.2 可選：torch.compile（加速 forward）

- 在建立 SAC 後對 `model.policy.features_extractor` 做 `torch.compile(...)`，可讓 4×CNN + Cross-Attention + MLP + fusion 的 forward 變快（通常約 1–2 成，依硬體與 PyTorch 版本而定）。
- 需 PyTorch 2+，且第一次會多一點編譯時間。
- 已加設定開關（如 `compile_policy`），預設關閉，需要時再開。

### 4.3 可選：關閉 Cross-Attention（用速度換表現）

- `use_cross_attention=False` 會少掉 5m/1d 的 cross-attention，forward 會變快，但可能影響策略品質，僅在「確定要衝 it/s」時可試。

---

## 5. GPU↔CPU 傳輸瓶頸（很多卡在這裡）

每步與每次 train 的資料都在 CPU 與 GPU 之間搬運，這本身就會卡住：

| 時點 | 方向 | 內容 | 說明 |
|------|------|------|------|
| **每 step predict** | CPU→GPU | 整份 obs Dict（64 envs × 各 key shape） | VecEnv 回傳 numpy，SB3 轉 tensor 並 `.to(device)` |
| **每 step predict** | GPU→CPU | actions（64 個 float） | 回傳給 `env.step()` 一定要 numpy，所以會 sync |
| **每 train** | CPU→GPU | sample 出的 batch（256 × obs） | buffer 在 CPU，`to_torch` 會把 batch 搬到 device |

因此「很多計算瓶頸都卡在 GPU 轉 CPU（或 CPU 轉 GPU）」是對的：  
不是算得慢，而是**搬資料**與**等 sync** 佔掉時間。  
我們能做的只有：**讓每次搬運更省、或稍微重疊**，無法在現有架構下完全拿掉這些傳輸。

### 5.1 已做／可做的傳輸優化

1. **Obs 保持 contiguous**  
   - 若 obs 的 numpy 是 contiguous，`torch.from_numpy(...).to(device)` 的複製較有效率。  
   - 在 **Observer** 回傳前對陣列做 `np.ascontiguousarray`（僅在尚未 contiguous 時），減少不連續造成的多餘拷貝。

2. **Replay buffer sample：non_blocking 搬到 GPU**  
   - 在 **OptimizedDictReplayBuffer** 的 sample 路徑，對 obs/next_obs 做 `.to(device, non_blocking=True)`（僅在 device 為 cuda 時）。  
   - 傳輸非同步，有機會與後續一點 CPU 工作重疊，減少「空等傳輸」的時間。

3. **Predict 路徑的 CPU↔GPU**  
   - obs 來自 SubprocVecEnv（一定在 CPU），predict 後 actions 必須給 env（一定要回 CPU）。  
   - 這段在 SB3 內部，我們無法改「要不要傳」，只能從 **obs 體積**（例如縮小 window/特徵）或 **env 端保證 contiguous** 讓傳輸本身快一點。

---

## 6. 沒有冗餘的結論

- **沒有**同一 step 內對同一 obs 做兩次 predict 或兩次 feature forward。
- **沒有**在 callback 或 buffer 裡對同一筆 obs 再跑一次 policy/critic。
- 瓶頸就是：**每一 step 那「一次」predict 的整條路徑**，以及 **GPU↔CPU 的傳輸與 sync**。  
  優化方向：**減少路徑上的不必要工作**（.float 條件化、buffer copyto）、**加速 forward**（compile）、以及 **減輕傳輸**（obs contiguous、buffer sample 時 non_blocking）。
