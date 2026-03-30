"""讀取 Phase A/B TensorBoard 日誌並輸出摘要。可指定目錄路徑。

若某個 scalar 不存在（例如舊 run 沒有 cost_risk_dense_sum_mean、或尚無 cost_fric/*），該欄位輸出為空，不報錯。
"""
import argparse
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

KEYS = [
    "episode_stats/log_return_sum_mean",
    "episode_stats/final_balance_mean",
    "episode_stats/cost_risk_sum_mean",
    "episode_stats/cost_risk_dense_sum_mean",
    "episode_stats/auxiliary_main_ratio",
    "episode_stats/trade_count_mean",
    "episode_stats/total_fees_mean",
    "episode_stats/fees_profit_ratio_mean",
    # Phase B：reward 合成中的摩擦懲罰項、摩擦細項（新 run；舊 log 無則空欄）
    "reward_decomp/penalty_fee_mean",
    "cost_fric/used_mean",
    "cost_fric/fee_component_mean",
    "cost_fric/activity_component_mean",
    "cost_fric/extreme_component_mean",
    "cost_fric/lambda_fee_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Phase A/B TensorBoard logs and print CSV.")
    parser.add_argument(
        "path",
        nargs="?",
        default="logs/phase_A_r000003",
        help="TensorBoard log 目錄（預設: logs/phase_A_r000003）",
    )
    args = parser.parse_args()
    path = args.path.strip()
    if not os.path.isdir(path):
        print(f"Error: directory not found: {path}", file=sys.stderr)
        sys.exit(1)

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    # 每個 key 的 (step -> value) 對照，缺的 key 用空 dict
    by_key: dict[str, dict[int, float]] = {}
    steps_primary: list[int] = []

    for k in KEYS:
        s = ea.Scalars(k)
        if s is None:
            s = []
        by_key[k] = {int(e.step): float(e.value) for e in s}
        if k == KEYS[0] and s:
            steps_primary = sorted(by_key[k].keys())

    if not steps_primary:
        print(f"No scalars found in {path}", file=sys.stderr)
        sys.exit(1)

    print(
        "step,log_return_mean,final_balance_mean,cost_risk_mean,"
        "cost_risk_dense_mean,auxiliary_main_ratio,trade_count_mean,"
        "total_fees_mean,fees_profit_ratio,penalty_fee_mean,"
        "cf_used_mean,cf_fee_component_mean,cf_activity_component_mean,"
        "cf_extreme_component_mean,cf_lambda_fee_mean"
    )
    for step in steps_primary:
        row = [str(step)]
        for k in KEYS:
            val = by_key.get(k, {}).get(step)
            if val is not None:
                row.append(str(round(val, 6)))
            else:
                row.append("")
        print(",".join(row))


if __name__ == "__main__":
    main()
