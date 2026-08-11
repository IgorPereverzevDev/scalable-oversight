"""Freeze the exact monitor prompts produced by v0.1.1 trajectories.

    python scripts/freeze_monitor_inputs.py outputs/trajectories.json --write
    python scripts/freeze_monitor_inputs.py outputs/trajectories.json   # verify

The monitor's input is the one channel through which a port can change verdicts
without touching the prompt, the model or the generation parameters. Freezing it
turns "the render looks fine" into a check that fails loudly.

Golden files are stored per prompt version, so changing the prompt in v0.2 adds a
directory instead of overwriting the evidence for v0.1.1.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cerberus.adapters.legacy import load_trajectories  # noqa: E402
from cerberus.core.monitors import BASELINE_MONITOR  # noqa: E402
from cerberus.evals.dataset import SPECS_BY_ID  # noqa: E402

GOLDEN_ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories")
    parser.add_argument("--write", action="store_true", help="create missing goldens")
    parser.add_argument("--force", action="store_true", help="overwrite existing goldens")
    args = parser.parse_args()

    target = GOLDEN_ROOT / f"monitor_inputs_{BASELINE_MONITOR.prompt}"
    target.mkdir(parents=True, exist_ok=True)

    mismatches = 0
    for traj in load_trajectories(args.trajectories):
        spec = SPECS_BY_ID.get(traj.task_id)
        if spec is None:
            print(f"SKIP {traj.task_id}: not in the current dataset")
            continue

        prompt = BASELINE_MONITOR.build_prompt(traj, spec["user_prompt"])
        path = target / f"{traj.task_id}.txt"

        if not path.exists():
            if args.write:
                path.write_text(prompt, encoding="utf-8")
                print(f"WROTE {path.name}")
            else:
                print(f"MISSING {path.name} (run with --write)")
                mismatches += 1
            continue

        if path.read_text(encoding="utf-8") == prompt:
            continue

        if args.force and args.write:
            path.write_text(prompt, encoding="utf-8")
            print(f"OVERWROTE {path.name}")
        else:
            print(f"MISMATCH {path.name} — monitor input changed")
            mismatches += 1

    if mismatches:
        print(f"\n{mismatches} file(s) differ. Either revert the change, or accept it")
        print("with --write --force and treat the results as a new condition.")
        return 1

    print(f"monitor inputs match the frozen baseline in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
