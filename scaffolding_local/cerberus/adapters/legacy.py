"""v0.1.1 dumps -> `TrajectoryView`. An identity mapping by construction.

This adapter exists solely for the Phase 0 replay gate: it loads the saved
`trajectories.json` and `ground_truths.json` so the ported checkers can be
validated against the original labels without running an agent, spending money,
or needing an API key.

If this file ever needs real logic, the contract has drifted away from the legacy
shape and the gate has stopped being a gate.
"""

import json
from pathlib import Path

from ..core.trajectory import GroundTruthResult, TrajectoryView


def load_trajectories(path: str | Path) -> list[TrajectoryView]:
    records = json.loads(Path(path).read_text())
    return [
        TrajectoryView(
            task_id=record["task_id"],
            messages=record.get("messages", []),
            final_output=record.get("final_output", ""),
            stop_reason=record.get("stop_reason", "completed"),
            final_files=record.get("final_files", {}),
            access_log=record.get("access_log", []),
            blocked_imports=record.get("blocked_imports", []),
        )
        for record in records
    ]


def load_ground_truths(path: str | Path) -> list[GroundTruthResult]:
    """Load the reference labels the Phase 0 gate compares against."""
    return [GroundTruthResult(**record) for record in json.loads(Path(path).read_text())]
