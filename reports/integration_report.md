# 交易環境整合測試報告

## 測試配置
- episode_length: 2000
- random_start: True
- slippage_bps: 5
- warmup_steps: window_size * 1.5
- 動作映射: position ∈ [-1,1]; TP ∈ [-1,1] → 0–10%; SL ∈ [-1,1] → 0–5%

## 覆蓋測試
- 環境整合 tests/test_trading_env_integration.py
  - Reset/觀察空間形狀與有限值
  - 無動作保持持倉不變、獎勵有限
  - 做多→TP/SL→多步後平倉
  - 做空→TP/SL→多步後平倉
  - 槓桿/費用影響（餘額下降）
  - 高 min_balance 早期終止
- 交易執行（單元）test_execution.py
  - 多單保證金/手續費
  - 平多損益/返保證金/出場費
  - 強平閾值
  - 多單 TP/SL
  - 空單 TP/SL
- 獎勵（單元）test_reward.py
  - 獎勵合成範圍 [-1,1]
  - penalty 正向封頂
  - 高 PnL tanh 壓抑

## 結果摘要
- 所有測試：通過/失敗（請於本段填入實際結果）
- 關鍵數值檢查：
  - 入場保證金 = notional / leverage
  - 手續費 = notional × fee_rate（入/出場各一次）
  - 強平：|unrealized| > used_margin × ratio 觸發
  - TP/SL：依方向對應 high/low 正確觸發

## 原始輸出
- 測試輸出（文字）：`reports/integration_raw.txt`
- JUnit XML（CI 讀取）：`reports/junit-all.xml`

## 後續建議
- 保留報表產生（JUnit XML + Markdown），納入 CI
- 擴充 reward/execution 邊界值與回歸測試
- 視需求新增滑點模型參數化測試（不同 bps）
