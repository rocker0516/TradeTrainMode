#!/bin/bash
# 模型評估範例腳本

# ============================================================
# 基本評估（最簡潔）
# ============================================================
python Train/evaluate.py \
  --model training_results/sac_btc_3m/sac_btc_3m_model.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 10 \
  --deterministic

# ============================================================
# 完整評估
# ============================================================
python Train/evaluate.py \
  --model training_results/sac_btc_3m/sac_btc_3m_model.zip \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --episode_steps 500 \
  --deterministic \
  --leverage 5 \
  --position_scale 0.5 \
  --output evaluation_results/btc_test

# ============================================================
# 不同時間段評估
# ============================================================

# Q1 2024
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 30 \
  --deterministic \
  --output eval/q1_2024

# Q2 2024
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --start_date 20240401 \
  --end_date 20240630 \
  --episodes 30 \
  --deterministic \
  --output eval/q2_2024

# ============================================================
# 多幣種評估
# ============================================================

# BTC 評估
python Train/evaluate.py \
  --model models/sac_btc.zip \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --output eval/btc

# ETH 評估
python Train/evaluate.py \
  --model models/sac_eth.zip \
  --csv Data/ETHUSDT_futures_volume_5years_5min.csv \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --output eval/eth

# SOL 評估
python Train/evaluate.py \
  --model models/sac_sol.zip \
  --csv Data/SOLUSDT_futures_volume_5years_5min.csv \
  --start_date 20240101 \
  --end_date 20240331 \
  --episodes 50 \
  --deterministic \
  --output eval/sol

# ============================================================
# 參數說明
# ============================================================
# 必需參數：
#   --model: 模型路徑（.zip文件）
#   --start_date: 測試起始日期
#   --end_date: 測試結束日期
#
# 常用可選參數：
#   --episodes: 測試回合數（默認10）
#   --deterministic: 使用確定性策略（推薦加上）
#   --episode_steps: 每回合最大步數（0=不限）
#   --output: 輸出目錄
#
# 環境參數（應與訓練時一致）：
#   --leverage: 槓桿倍數（默認5）
#   --position_scale: 倉位上限（默認0.5）
#   --initial_balance: 初始資金（默認10000）
# ============================================================

