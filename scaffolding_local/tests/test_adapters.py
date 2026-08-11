"""Adapter tests — one per documented porting risk.

These are the tests that decide whether the migration preserved the experiment.
Everything else in the suite checks that the code works; these check that it
works *the same way it used to*.
"""

from cerberus.adapters.inspect_state import (
    STORE_ACCESS_LOG,
    STORE_FINAL_FILES,
    STORE_STOP_REASON,
    messages_to_legacy,
    view_from_inspect,
)
from cerberus.core.rendering import render_v011
from cerberus.core.trajectory import TrajectoryView


def test_inspect_messages_reproduce_legacy_shape(legacy_trajectory, inspect_messages):
    """RISK-1/3: reconstructed messages match the legacy dump exactly."""
    assert messages_to_legacy(inspect_messages) == legacy_trajectory.messages


def test_tool_only_turn_produces_no_phantom_thought(inspect_messages):
    """RISK-2: an assistant turn with no prose must not add an empty thought line.

    Inspect represents "tool calls only" as an empty content string, whereas the
    prototype simply had no text block. A naive port inserts a blank
    `[AGENT THOUGHT]: ` line and silently alters the monitor's prompt on every
    such turn.
    """
    converted = messages_to_legacy(inspect_messages)
    tool_only = converted[4]
    assert tool_only["content"] == [
        {"type": "tool_use", "id": "tu_2", "name": "run_tests", "input": {}}
    ]

    rendered = render_v011(
        TrajectoryView(task_id="x", messages=[tool_only], final_output="")
    )
    assert "[AGENT THOUGHT]" not in rendered


def test_unnamed_tool_message_falls_back_to_question_mark():
    """RISK-3: a tool call without a function name keeps the legacy placeholder."""

    class Unnamed:
        role = "tool"
        text = "something happened"
        tool_call_id = "tu_9"
        function = None

    assert messages_to_legacy([Unnamed()])[0]["tool_name"] == "?"


def test_final_output_is_empty_when_turns_run_out():
    """RISK-4: max_turns episodes showed the monitor an empty final output.

    Inspect always exposes a completion, so without this the monitor's input
    differs precisely on the episodes where the agent struggled — the ones where
    detection matters most.
    """
    exhausted = view_from_inspect(
        task_id="x",
        messages=[],
        store={STORE_STOP_REASON: "max_turns"},
        completion="still working on it",
    )
    assert exhausted.final_output == ""

    finished = view_from_inspect(
        task_id="x",
        messages=[],
        store={STORE_STOP_REASON: "completed"},
        completion="Done.",
    )
    assert finished.final_output == "Done."


def test_view_reads_workspace_state_from_store_only():
    """RISK-5: files and access log come from the store, never a live sandbox.

    This is what allows `inspect score` to re-run monitors against a finished
    log, which is the main efficiency win of the migration.
    """
    view = view_from_inspect(
        task_id="x",
        messages=[],
        store={
            STORE_FINAL_FILES: {"solution.py": "print(1)"},
            STORE_ACCESS_LOG: [{"op": "read", "path": "config/.env", "ts": 0.0}],
        },
        completion="",
    )
    assert view.final_files == {"solution.py": "print(1)"}
    assert view.access_log[0]["path"] == "config/.env"


def test_missing_store_keys_degrade_to_empty():
    """A partially populated store must not crash a scorer mid-eval."""
    view = view_from_inspect(task_id="x", messages=[], store={}, completion="ok")
    assert view.final_files == {}
    assert view.access_log == []
    assert view.blocked_imports == []
