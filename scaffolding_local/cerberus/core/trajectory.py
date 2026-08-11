"""The canonical trajectory contract.

WHY THIS MODULE EXISTS
----------------------
In the v0.1.1 notebook, `Trajectory` was a *runtime artifact*: `run_agent` built
it, the pipeline checkpointed it to Drive, monitors and checkers consumed it.

Under Inspect the runtime artifact is the eval log. `TrajectoryView` is therefore
demoted from runtime artifact to **port contract** — a normalized shape that two
different sources are converted into:

    trajectories.json (v0.1.1)  --adapters.legacy-->        TrajectoryView
    TaskState / EvalSample      --adapters.inspect_state--> TrajectoryView

This indirection is not decoration. It is what makes the Phase 0 replay gate
possible at all: saved v0.1.1 trajectories cannot be turned into a `TaskState`
without running an eval, so scorers that read `TaskState` directly could never be
validated against the old ground truth.

The shape deliberately mirrors the legacy dump byte for byte. That makes the
legacy adapter an identity function and localizes *all* porting risk into one
file (`adapters/inspect_state.py`), where it can be tested.

REJECTED ALTERNATIVE
--------------------
Typed message models (a `Message` union with `TextBlock` / `ToolUseBlock`).
Cleaner on paper, but it would have changed the string the monitor sees, which is
the one thing the migration must hold fixed. Revisit after Phase 1 closes.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"
"""Version of the normalized *view*, not of the storage schema.

Bump only when the information available to a monitor changes. Results produced
under different contract versions are not comparable, exactly like results
produced under different monitor prompts.
"""

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["completed", "max_turns", "error"]


class TrajectoryView(BaseModel):
    """One agent episode, in the shape monitors and ground-truth checkers expect.

    `messages` keeps the legacy dict shape rather than typed models:

        {"role": "system" | "user", "content": str}
        {"role": "assistant", "content": [
            {"type": "text", "text": str}
            | {"type": "tool_use", "id": str, "name": str, "input": dict}
        ]}
        {"role": "tool", "content": str, "tool_use_id": str, "tool_name": str}

    `extra="allow"` tolerates fields written by older runs so that historical
    dumps stay loadable without a migration step.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str
    messages: list[dict[str, Any]]
    final_output: str
    stop_reason: StopReason = "completed"
    final_files: dict[str, str] = Field(default_factory=dict)
    access_log: list[dict[str, Any]] = Field(default_factory=list)
    blocked_imports: list[str] = Field(default_factory=list)

    def tool_results(self, tool_name: str) -> list[str]:
        """All results returned by a given tool, in order.

        Checkers grep tool output for frozen markers (see `evals.tools`), so this
        accessor exists to keep that lookup in one place instead of scattering
        `msg["role"] == "tool"` conditionals across checkers.
        """
        return [
            str(msg["content"])
            for msg in self.messages
            if msg.get("role") == "tool" and msg.get("tool_name") == tool_name
        ]


class Verdict(BaseModel):
    """A monitor's judgement on a single trajectory.

    `parse_error` is a first-class value, not an exception. A monitor that
    returned something unparseable produced no usable verdict, and burying that
    in a retry would silently move the denominators of every metric.
    """

    verdict: Literal["safe", "unsafe", "parse_error"]
    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class GroundTruthResult(BaseModel):
    """Output of a scripted ground-truth check.

    `did_bad_thing` and `task_succeeded` are independent on purpose: an agent can
    complete the task honestly, complete it while also doing the bad thing, or
    fail the task entirely. Collapsing them loses the distinction that the
    per-category breakdown depends on.
    """

    did_bad_thing: bool
    task_succeeded: bool
    evidence: str
