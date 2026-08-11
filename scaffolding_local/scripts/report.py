"""Print metrics for an eval log, optionally gated against a frozen baseline.

    python scripts/report.py logs/2026-08-04_cerberus.eval
    python scripts/report.py logs/<run>.eval --baseline baselines/v0.1.1.json
    python scripts/report.py logs/<run>.eval --save baselines/v0.1.1.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai.log import read_eval_log  # noqa: E402

from cerberus.analysis.metrics import (  # noqa: E402
    compare_to_baseline,
    format_report,
    metrics_from_log,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--baseline", help="frozen metrics JSON to compare against")
    parser.add_argument("--save", help="write these metrics out as a baseline")
    args = parser.parse_args()

    metrics = metrics_from_log(read_eval_log(args.log))
    print(format_report(metrics))

    print(f"\n{'task':48s} {'category':13s} {'bad':5s} {'verdict':11s} outcome")
    for episode in metrics.per_episode:
        print(
            f"{episode.task_id:48s} {episode.category:13s} "
            f"{str(episode.did_bad_thing):5s} {episode.monitor_verdict:11s} "
            f"{episode.outcome}"
        )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(metrics.model_dump_json(indent=2))
        print(f"\nbaseline written to {args.save}")

    if args.baseline:
        print("\nbaseline comparison:")
        failed = False
        for name, status, detail in compare_to_baseline(metrics, args.baseline):
            print(f"  [{status:4s}] {name:28s} {detail}")
            failed |= status == "FAIL"
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
