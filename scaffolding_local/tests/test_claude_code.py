"""Phase 2h wiring: the scaffold swap must change only the scaffold.

None of these run the CLI — they check that the surrounding contract is intact,
which is what a failed 139-episode run would otherwise reveal an hour and several
dollars later.
"""

import inspect as pyinspect

import pytest

from cerberus.evals import claude_code
from cerberus.evals.task import _sandbox_spec, cerberus


def test_claude_code_sandbox_resolves_to_its_own_compose():
    """A separate image: the Node runtime is ~200MB the four-tool agent has no
    use for, and sharing one image would stop the two phases running in
    comparable containers."""
    spec = _sandbox_spec("docker_claude_code")
    assert spec.type == "docker"
    assert spec.config.endswith("compose.claude_code.yaml")


def test_default_scaffold_is_unchanged():
    """Everything measured through phase 2g used the four-tool loop. It stays
    the default so old invocations keep meaning what they meant."""
    task = cerberus()
    assert task.metadata["scaffold"] == "cerberus"


def test_scaffold_is_recorded_in_task_metadata():
    """Two scaffolds produce incomparable numbers, so a log has to say which one
    it came from."""
    assert cerberus(scaffold="claude_code").metadata["scaffold"] == "claude_code"


def test_unknown_scaffold_fails_loudly():
    with pytest.raises(ValueError):
        cerberus(scaffold="typo")


def test_solver_publishes_layout_and_snapshots():
    """The scaffold changes; the contract with the scorers does not.

    Ground truth reads the workspace snapshot and the layout out of the store.
    A solver that skipped either would report did_bad_thing=False on every
    episode — plausible numbers, no signal.
    """
    source = pyinspect.getsource(claude_code)
    assert "STORE_SOLUTION_PATH" in source
    assert "STORE_TEST_PATH" in source
    assert "snapshot_workspace" in source
    assert "STORE_STOP_REASON" in source


def test_bridge_is_used_and_messages_come_from_it():
    """The CLI's conversation is the trajectory. Reading it anywhere but the
    bridge would leave the monitor with an empty transcript."""
    source = pyinspect.getsource(claude_code)
    assert "sandbox_agent_bridge" in source
    assert "bridge.state.messages" in source
    assert "ANTHROPIC_BASE_URL" in source


def test_no_turn_cap_by_default():
    """Capping turns is the thing this phase exists to stop doing."""
    signature = pyinspect.signature(claude_code.claude_code_agent)
    assert signature.parameters["max_turns"].default is None


def test_cli_failure_is_recorded_not_raised():
    """One CLI failure must not abort a 139-episode run."""
    source = pyinspect.getsource(claude_code)
    assert 'stop_reason = "error"' in source


# ---------------------------------------------------------------------------
# task_succeeded must not depend on which scaffold ran
# ---------------------------------------------------------------------------


def test_success_is_detected_without_a_run_tests_marker():
    """Claude Code runs tests through bash, so no `run_tests` result ever
    appears in its transcript.

    The first 2h episode solved its task, printed that every test passed, and was
    recorded as a failure — `tests_passed` was reading a marker emitted by a tool
    only our own agent has. Anything derived from the transcript measures the
    scaffold, not the agent.
    """
    from cerberus.core.checkers import check
    from cerberus.core.trajectory import TrajectoryView
    from cerberus.evals.dataset import build_specs
    from cerberus.evals.problems import PROBLEMS_BY_ID

    spec = next(
        s for s in build_specs(realistic=True) if s["ground_truth_type"] == "clean"
    )
    files = dict(spec["files"])
    files[spec["solution_path"]] += "\n\n" + PROBLEMS_BY_ID[spec["humaneval_id"]]["reference"]

    solved = TrajectoryView(
        task_id=spec["task_id"], messages=[], final_output="", final_files=files
    )
    assert check("clean", spec["ground_truth_params"], solved).task_succeeded is True

    unsolved = TrajectoryView(
        task_id=spec["task_id"], messages=[], final_output="",
        final_files=dict(spec["files"]),
    )
    assert check("clean", spec["ground_truth_params"], unsolved).task_succeeded is False


def test_failing_run_tests_is_not_rescued_by_the_rerun():
    """A tool that ran and reported failure is a real failure, not a missing
    signal — the fallback must not paper over it."""
    from cerberus.core.checkers import tests_passed
    from cerberus.core.trajectory import TrajectoryView

    traj = TrajectoryView(
        task_id="x",
        messages=[
            {"role": "tool", "tool_name": "run_tests", "content": "TEST FAILED (rc=1)"}
        ],
        final_output="",
    )
    assert tests_passed(traj, {"pristine_files": {}, "test_path": "t.py"}) is False