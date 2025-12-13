# Baseline Strategy Test Report

This report summarizes the performance of baseline strategies on the BTC/USDT Futures dataset (5 years, 5-minute intervals).

## Configuration
- **Dataset**: `Data/BTCUSDT_futures_volume_5years_5min.csv`
- **Initial Balance**: 10,000 USDT
- **Leverage**: 10x
- **Transaction Fee**: 0.04% (Taker)
- **Margin Mode**: Isolated
- **Min Balance**: 100 USDT (1% of Initial)
- **Liquidation**: Triggered if Maintenance Margin Ratio > 100% (or Equity <= Min Balance)

## Results

| Strategy | Profit % | MDD % | Final Equity | Steps Survived | Termination Reason |
|----------|----------|-------|--------------|----------------|--------------------|
| **No-Trade** | 0.00% | 0.00% | 10,000.00 | 524,736 (Full) | Data Exhausted |
| **Always-Long** | -99.04% | 99.71% | 95.62 | 31,684 | Balance Insufficient |
| **Always-Short** | -99.07% | 99.08% | 92.82 | 8,063 | Balance Insufficient |
| **SMA-200** | -99.04% | 99.04% | 96.15 | 33,596 | Balance Insufficient |

## Analysis

1.  **No-Trade**:
    -   Successfully ran through the entire dataset (approx. 5 years) with no errors.
    -   Validates the environment's data loading and step iteration logic.

2.  **Active Strategies (Always-Long/Short, SMA)**:
    -   **All busted early.**
    -   **Reason 1: High Leverage (10x)**: The default 10x leverage is extremely risky for automated strategies without sophisticated risk management. A 10% adverse price move leads to liquidation in Isolated Margin mode.
    -   **Reason 2: Constant Leverage Rebalancing**: The environment interprets `action=1.0` as "Target 10x Leverage". As equity fluctuates, the agent constantly adjusts position size to maintain this ratio.
        -   **Volatility Drag**: This forces "Buying High and Selling Low" to rebalance (e.g., if equity drops, position must be reduced -> Sell after drop).
        -   **Fee Erosion**: Continuous rebalancing incurs 0.04% fee frequently, draining capital.
    -   **Reason 3: Volatility**: BTC futures market contains multiple >10% drawdowns which are fatal for 10x leverage strategies.

## Conclusion
The environment correctly simulates the mechanics of trading (fees, pnl, liquidation). The baseline failures confirm that naive high-leverage strategies are not viable and that the RL agent faces a challenging task to learn profitable risk management.

