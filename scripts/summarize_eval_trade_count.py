"""從 Phase B eval JSON 彙總 episode_steps / episode_trade_count / execution_rate。

用於驗證：交易次數與 episode 步數強相關，因同一資料集下 episode 長度分佈固定。
用法: python scripts/summarize_eval_trade_count.py logs/phase_B_*_eval.json
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize episode_steps vs trade_count from Phase B eval JSON(s)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more *_eval.json file paths",
    )
    args = parser.parse_args()

    print("file,episodes,mean_steps,mean_trade_count,mean_exec_rate,reason")
    for path in args.paths:
        p = Path(path)
        if not p.is_file():
            print(f"{path},NOT_FOUND,,,,", file=sys.stderr)
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"{path},ERROR,{e},,,", file=sys.stderr)
            continue
        results = data.get("results") or []
        if not results:
            print(f"{path},0,,,,", file=sys.stderr)
            continue
        steps = [r.get("episode_steps", 0) for r in results]
        trades = [r.get("episode_trade_count", 0) for r in results]
        mean_steps = sum(steps) / len(steps)
        mean_trades = sum(trades) / len(trades)
        exec_rate = mean_trades / mean_steps if mean_steps > 0 else 0.0
        reason = data.get("reason", "")
        print(f"{p.name},{len(results)},{mean_steps:.0f},{mean_trades:.0f},{exec_rate:.4f},{reason}")


if __name__ == "__main__":
    main()
