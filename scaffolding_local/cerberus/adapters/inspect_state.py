"""Inspect -> `TrajectoryView`. The highest-risk module in the migration.

Every place where the port can silently change a number lives here, marked
RISK-n and pinned by a test in `tests/test_adapters.py`. The list is closed: if a
sixth risk appears, it belongs here too rather than in a scorer.

The functions take messages and a store mapping rather than a `TaskState` so that
they work in three situations: during an eval, during `inspect score` on a
finished log, and during offline analysis of an `EvalSample`.
"""

from typing import Any, Protocol

from ..core.trajectory import TrajectoryView

STORE_ACCESS_LOG = "cerberus:access_log"
STORE_BLOCKED_IMPORTS = "cerberus:blocked_imports"
STORE_FINAL_FILES = "cerberus:final_files"
STORE_STOP_REASON = "cerberus:stop_reason"
STORE_SOLUTION_PATH = "cerberus:solution_path"
STORE_TEST_PATH = "cerberus:test_path"
# Where the solution and the test file live inside the workspace. Constant until
# phase 2g: a realistic repository puts the target function in
# `<package>/<module>.py`, and every checker, tool and prompt that assumed
# "solution.py" has to be told otherwise. Carried in the store so tools can read
# it during the run without threading metadata through every call.


class _Message(Protocol):
    """The subset of an Inspect chat message this module relies on.

    Declared structurally so the adapter can be unit-tested with fakes and does
    not drag the Inspect runtime into the test path.
    """

    role: str


def _assistant_blocks(msg: Any) -> list[dict[str, Any]]:
    """Rebuild the legacy assistant block list from an Inspect message.

    RISK-1 — BLOCK ORDER.
    The notebook stored blocks in the order the Anthropic API returned them, as a
    single `content` list interleaving text and tool_use. Inspect splits these:
    prose lives in `.content`, calls live in `.tool_calls`, and their relative
    order is not preserved. Anthropic emits text before tool_use in practice, so
    that order is reconstructed. This is an approximation that holds for short
    trajectories and must be re-validated when real agent scaffolds start
    interleaving (Phase 2).

    RISK-2 — EMPTY TEXT.
    The notebook emitted an `[AGENT THOUGHT]` line per text block. When a model
    replied with tool calls only there was no text block, so no line. Inspect
    represents the same message with an empty string, and a naive port would add
    a blank `[AGENT THOUGHT]: ` line — silently changing the monitor's prompt on
    exactly those turns.

    Reasoning blocks are intentionally dropped: v0.1.1 monitors never saw them.
    Exposing them is a legitimate future experiment ("does seeing the agent's
    reasoning help or hurt the monitor?") and must be an explicit renderer
    variant, not a side effect of changing frameworks.
    """
    blocks: list[dict[str, Any]] = []
    content = getattr(msg, "content", "")

    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    else:
        for part in content:
            if getattr(part, "type", None) == "text":
                text = getattr(part, "text", "")
                if text:
                    blocks.append({"type": "text", "text": text})

    for call in getattr(msg, "tool_calls", None) or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function,
                "input": call.arguments,
            }
        )
    return blocks


def messages_to_legacy(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Inspect chat messages into the legacy message shape."""
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.role

        if role == "system":
            out.append({"role": "system", "content": msg.text})

        elif role == "user":
            out.append({"role": "user", "content": msg.text})

        elif role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(msg)})

        elif role == "tool":
            # RISK-3 — TOOL NAME.
            # `tests_passed()` and the renderer both key off tool_name. Inspect
            # calls it `function`; a failed call may leave it unset, which the
            # legacy format represented as "?".
            out.append(
                {
                    "role": "tool",
                    "content": msg.text,
                    "tool_use_id": msg.tool_call_id,
                    "tool_name": msg.function or "?",
                }
            )

    return out


def view_from_inspect(
    *,
    task_id: str,
    messages: list[Any],
    store: dict[str, Any],
    completion: str,
) -> TrajectoryView:
    """Build the port contract from Inspect state.

    RISK-4 — FINAL OUTPUT UNDER max_turns.
    The notebook only assigned `final_text` on the end_turn branch. An agent that
    exhausted its turn budget left it empty, and the monitor saw a bare
    `[FINAL OUTPUT]: `. Inspect always exposes a completion, so the old behaviour
    is restored explicitly — otherwise the monitor's input differs precisely on
    the episodes where the agent struggled.

    RISK-5 — SOURCE OF FILES AND ACCESS LOG.
    Both come from the sample store, never from a live sandbox. This is a hard
    requirement, not an optimisation: `inspect score` re-runs scorers against a
    finished log, when no container exists. Reading the sandbox here would make
    monitor re-evaluation impossible and forfeit the main benefit of the
    migration — run the agent once, run monitors many times.
    """
    stop_reason = store.get(STORE_STOP_REASON, "completed")

    return TrajectoryView(
        task_id=task_id,
        messages=messages_to_legacy(messages),
        final_output="" if stop_reason == "max_turns" else completion,
        stop_reason=stop_reason,
        final_files=store.get(STORE_FINAL_FILES) or {},
        access_log=store.get(STORE_ACCESS_LOG) or [],
        blocked_imports=store.get(STORE_BLOCKED_IMPORTS) or [],
    )