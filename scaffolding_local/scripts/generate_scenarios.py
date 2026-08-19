"""Generate, validate and store adversarial scenarios.

    python scripts/generate_scenarios.py payments 5 --model anthropic/claude-opus-4-5
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from cerberus.audit.corpus import EXPLORATION_DIR, save, yield_report  # noqa: E402
from cerberus.audit.generator import generate_scenarios  # noqa: E402
from cerberus.audit.validator import validate  # noqa: E402
from cerberus.evals.repos import TEMPLATES  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", choices=sorted(TEMPLATES))
    parser.add_argument("count", type=int)
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-5")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    template = TEMPLATES[args.repo]
    scenarios, rejects = await generate_scenarios(
        template, args.count, model=args.model, seed_offset=args.seed_offset
    )

    for reason in rejects:
        print(f"malformed: {reason}")

    results = [validate(s, template.files) for s in scenarios]
    accepted = [s for s, r in zip(scenarios, results) if r.accepted]
    save(accepted, EXPLORATION_DIR)

    print()
    print(yield_report(scenarios, results))
    print()
    for scenario, result in zip(scenarios, results):
        if not result.accepted:
            print(f"rejected {scenario.scenario_id}: {result.failed_checks}")
            for key, value in result.detail.items():
                print(f"    {key}: {str(value)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))