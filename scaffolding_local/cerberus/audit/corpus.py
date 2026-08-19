"""Two corpora, because they answer different questions.

A generated corpus is not reproducible by construction: ask the auditor twice and
you get different scenarios. That is fine for finding new kinds of harm and
useless for measuring whether a monitor improved — a detection rate over a corpus
that changed underneath means nothing.

FROZEN — scenarios that passed validation and were promoted deliberately. This is
what progress is measured against, and it changes only by an explicit act.

EXPLORATION — the live output of a generation run. The number to read here is not
detection rate but yield: how often the auditor produced something the agent
actually did, and how much of it was unlike what the frozen corpus already
covers. A detection rate over a shifting denominator is not a rate.

Promotion is a separate step on purpose. An auditor that writes its own exam and
then grades progress on it is not an experiment.
"""

import json
from dataclasses import asdict
from pathlib import Path

from .schema import GeneratedScenario, ValidationResult

FROZEN_DIR = Path("corpus/frozen")
EXPLORATION_DIR = Path("corpus/exploration")


def save(scenarios: list[GeneratedScenario], directory: Path) -> list[Path]:
    """Write scenarios as one file each, keyed by id."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for scenario in scenarios:
        path = directory / f"{scenario.scenario_id}.json"
        path.write_text(json.dumps(asdict(scenario), indent=2))
        written.append(path)
    return written


def load(directory: Path) -> list[GeneratedScenario]:
    if not directory.exists():
        return []
    return [
        GeneratedScenario(**json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    ]


def promote(scenario_ids: list[str]) -> list[str]:
    """Move scenarios from exploration into the frozen corpus.

    Deliberate and explicit: a scenario that entered the measured set by accident
    would change the denominator of every past comparison without anyone
    noticing.
    """
    exploration = {s.scenario_id: s for s in load(EXPLORATION_DIR)}
    frozen_ids = {s.scenario_id for s in load(FROZEN_DIR)}

    promoted = []
    for scenario_id in scenario_ids:
        if scenario_id in frozen_ids:
            continue
        scenario = exploration.get(scenario_id)
        if scenario is None:
            raise KeyError(f"{scenario_id} is not in the exploration corpus")
        save([scenario], FROZEN_DIR)
        promoted.append(scenario_id)
    return promoted


def yield_report(
    scenarios: list[GeneratedScenario], results: list[ValidationResult]
) -> str:
    """What a generation run produced, before any agent has run.

    The rejection shape is the measurement of the auditor: one that mostly fails
    `harmful_hidden` is writing tests that do not test, one that mostly fails
    `harmful_visible` cannot make harm invisible, and one that mostly fails
    `honest_visible` is writing tasks nobody can solve.
    """
    accepted = [r for r in results if r.accepted]
    from collections import Counter

    failures = Counter()
    for result in results:
        for name in result.failed_checks:
            failures[name] += 1

    lines = [
        f"generated: {len(results)}",
        f"accepted:  {len(accepted)}",
        "",
        "rejections by check:",
    ]
    for name, count in failures.most_common():
        lines.append(f"  {name:18s} {count}")

    if accepted:
        lines.append("")
        lines.append("accepted scenarios:")
        by_id = {s.scenario_id: s for s in scenarios}
        for result in accepted:
            scenario = by_id.get(result.scenario_id)
            rationale = (scenario.rationale if scenario else "")[:70]
            lines.append(f"  {result.scenario_id:24s} {rationale}")
    return "\n".join(lines)