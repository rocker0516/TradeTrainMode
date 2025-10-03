#!/bin/bash
# 訓練腳本範例 - 使用新的參數計算方式
# 總步數 = vec_envs × episodes × episode_steps

# ============================================================
# 快速測試（約 1 分鐘）
# ============================================================
# 總步數 = 2 × 10 × 50 = 1,000
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230131 \
  --vec_envs 2 \
  --episodes 10 \
  --episode_steps 50 \
  --logdir training_results/quick_test

# ============================================================
# 標準訓練（約 30 分鐘）
# ============================================================
# 總步數 = 8 × 100 × 500 = 400,000
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230331 \
  --vec_envs 8 \
  --episodes 100 \
  --episode_steps 500 \
  --leverage 3 \
  --position_scale 0.3 \
  --logdir training_results/standard_train

# ============================================================
# 大規模訓練（約 4 小時）
# ============================================================
# 總步數 = 16 × 500 × 1000 = 8,000,000
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20220101 \
  --end_date 20231231 \
  --vec_envs 16 \
  --episodes 500 \
  --episode_steps 1000 \
  --leverage 5 \
  --position_scale 0.5 \
  --initial_balance 10000 \
  --logdir training_results/large_scale_train

# ============================================================
# 高頻策略訓練（短回合，多回合）
# ============================================================
# 總步數 = 16 × 300 × 288 = 1,382,400
# episode_steps=288 = 24小時（5分鐘K線）
python Train/train.py \
  --csv Data/BTCUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230630 \
  --vec_envs 16 \
  --episodes 300 \
  --episode_steps 288 \
  --leverage 3 \
  --position_scale 0.2 \
  --logdir training_results/high_freq_strategy

# ============================================================
# 多幣種訓練範例
# ============================================================

# ETH 訓練
python Train/train.py \
  --csv Data/ETHUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230630 \
  --vec_envs 16 \
  --episodes 200 \
  --episode_steps 500 \
  --logdir training_results/eth_train

# SOL 訓練
python Train/train.py \
  --csv Data/SOLUSDT_futures_volume_5years_5min.csv \
  --start_date 20230101 \
  --end_date 20230630 \
  --vec_envs 16 \
  --episodes 200 \
  --episode_steps 500 \
  --logdir training_results/sol_train

# ============================================================
# 參數說明
# ============================================================
# --vec_envs: 並行環境數（建議：CPU核心數的1-2倍）
# --episodes: 每個環境訓練的回合數（必需）
# --episode_steps: 每回合最大步數（必需）
# 
# 總訓練步數 = vec_envs × episodes × episode_steps
# 
# 範例計算：
# - 16 × 100 × 500 = 800,000 步
# - 8 × 200 × 400 = 640,000 步
# - 32 × 50 × 600 = 960,000 步
# ============================================================

