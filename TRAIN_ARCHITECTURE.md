# Training Architecture: SAC-Lagrangian + CNN

## Overview
This system implements a **Soft Actor-Critic (SAC)** agent with **Lagrangian Constraint Handling** (Safe RL) to train a trading agent. The agent consumes a multi-modal observation space consisting of time-series price data and a structured state vector.

## Components

### 1. Observation Space
- **Price Sequence (`price_seq`)**: 
  - Shape: `(Window_Size=288, Features=6)`
  - Features: Normalized Returns, Relative Range, Relative Body, Log Volume, Volatility Regime, Spread.
  - Encoder: **1D CNN** (3 layers of Conv1d + Pooling) -> Flatten -> Dense -> Embedding (128 dim).
  
- **State Vector (`state_vector`)**:
  - Shape: `(State_Dim=20)`
  - Features: Position info, Account metrics (Equity, Drawdown), Market regime flags, Time info.
  - Encoder: **Identity** (Directly concatenated with Price Embedding).

### 2. Neural Network Architecture
- **Fusion**: Concatenates Price Embedding (128) + State Vector (20) = 148 dimension feature vector.
- **Actor (Policy)**: 
  - Input: Fusion Vector.
  - Hidden: 256 -> 256.
  - Output: Mean, LogStd (for TanhGaussian Policy).
- **Critic (Q-Function)**:
  - Input: Fusion Vector + Action.
  - Hidden: 256 -> 256.
  - Output: Q-Value (Scalar).
  - Structure: Double Q-Learning (Critic 1 & Critic 2).
- **Cost Critic (Safety)**:
  - Similar architecture to Critic but predicts **Expected Cost**.
  - Used for Lagrangian constraint updates.

### 3. SAC-Lagrangian Agent
- **Objective**: Maximize Entropy-Augmented Reward subject to Expected Cost <= Limit.
- **Algorithm**:
  1. Update Critic (Reward) using Bellman Error.
  2. Update Cost Critic (Safety) using Bellman Error.
  3. Update Actor to maximize: `Q_Reward - lambda * Q_Cost + alpha * Entropy`.
  4. Update Alpha (Entropy Temp) automatically.
  5. Update Lambda (Lagrangian Multiplier) via dual gradient ascent: `lambda += lr * (Q_Cost - Limit)`.

### 4. Cost Function
- Defined in `Train/cost.py`.
- Modular design allowing different cost definitions (e.g., Drawdown, Transaction Fees, Risk Metrics).
- Current: `CombinedCostCalculator` (extensible).

## Usage
Run training via module:
```bash
python -m Train.train
```

View progress:
```bash
tensorboard --logdir runs/sac_lagrangian_cnn
```
