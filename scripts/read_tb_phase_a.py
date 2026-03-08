"""讀取 Phase A/B TensorBoard 日誌並輸出摘要。可指定目錄路徑。"""
import argparse
import sys

from tensorboard.backend.event_processing import event_accumulator

KEYS = [
    "episode_stats/log_return_sum_mean",
    "episode_stats/final_balance_mean",
    "episode_stats/cost_risk_sum_mean",
    "episode_stats/auxiliary_main_ratio",
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

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    scalars = ea.Scalars(KEYS[0])
    if not scalars:
        print(f"No scalars found in {path}", file=sys.stderr)
        sys.exit(1)

    print("step,log_return_mean,final_balance_mean,cost_risk_mean,auxiliary_main_ratio")
    for i in range(len(scalars)):
        step = scalars[i].step
        row = [str(step)]
        for k in KEYS:
            s = ea.Scalars(k)
            row.append(str(round(s[i].value, 6)) if i < len(s) else "")
        print(",".join(row))


if __name__ == "__main__":
    main()
