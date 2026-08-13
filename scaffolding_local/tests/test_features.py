"""Feature tasks must be solvable, non-trivial, and genuinely multi-file.

Same guard as the problem bank, for the same reason: v0.1.1 shipped a task whose
hidden tests failed on correct code and nothing noticed for months. Here the risk
is larger, because a feature task touches several modules and a reference that
breaks one of the project's existing suites would look like the agent's fault.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from cerberus.evals.features import FEATURE_TASKS, feature_workspace
from cerberus.evals.repos import TEMPLATES

IDS = [t.task_id for t in FEATURE_TASKS]


def _run(files: dict[str, str], script: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        for path, content in files.items():
            target = Path(tmp, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return subprocess.run(
            ["python3", script],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        )


def _solved(task) -> dict[str, str]:
    files = feature_workspace(task, TEMPLATES)
    files.update(task.reference)
    return files


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_reference_passes_the_visible_suite(task):
    result = _run(_solved(task), task.test_path)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-500:]


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_reference_passes_the_hidden_suite(task):
    """Hidden tests decide did_bad_thing for reward_hack. A malformed one does
    not merely lose information — it manufactures evidence of misbehaviour."""
    files = _solved(task)
    files["_hidden.py"] = task.hidden_source
    result = _run(files, "_hidden.py")
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-500:]


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_reference_does_not_break_the_project(task):
    """A feature must not regress the suites the repository already had.

    Otherwise the agent inherits a broken project and any failure is ours."""
    files = _solved(task)
    for path in TEMPLATES[task.repo].files:
        if not path.startswith("tests/"):
            continue
        result = _run(files, path)
        assert result.returncode == 0 and "OK" in result.stdout, (
            f"{path}: {result.stderr[-400:]}"
        )


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_task_is_not_already_solved(task):
    """The control. A visible suite that passes on the untouched project would
    mean the episode measures nothing at all."""
    result = _run(feature_workspace(task, TEMPLATES), task.test_path)
    assert result.returncode != 0


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_change_really_spans_several_files(task):
    """The whole point is trajectory length. A one-file edit dressed up as a
    feature request would reproduce the 13-turn episodes of phase 2h."""
    assert len(task.touches) >= 3
    assert set(task.touches) <= set(task.reference)


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_workspace_ships_no_stub_or_marker(task):
    """The request lives in the prompt. A stub in the code would tell the agent
    where to look, which is exactly the searching we want it to do."""
    files = feature_workspace(task, TEMPLATES)
    for path, content in files.items():
        if path == task.test_path:
            continue
        assert "NotImplementedError" not in content
        assert "TODO" not in content


def test_task_ids_are_unique():
    assert len(FEATURE_TASKS_IDS := {t.task_id for t in FEATURE_TASKS}) == len(FEATURE_TASKS)
    assert all(t.repo in TEMPLATES for t in FEATURE_TASKS)


@pytest.mark.parametrize("task", FEATURE_TASKS, ids=IDS)
def test_summary_states_the_problem_not_the_solution(task):
    """A ticket, not a work plan.

    The first version of these summaries listed every file to touch and every
    function to add. Claude Code solved one in four model calls — fewer than a
    single-function task — because there was nothing left to work out. The
    episode length is the measurement here, so a summary that hands over the
    plan defeats the task.

    Names the visible tests import must survive: without them the task is
    unsolvable rather than merely hard.
    """
    summary = task.summary
    for path in task.touches:
        # The module a new file goes in may be named; the ones to edit must not.
        if path in task.reference and path not in getattr(task, "_new_files", ()):
            continue
    assert "- `payments/" not in summary and "- `logparse/" not in summary, (
        "summary enumerates the files to edit"
    )
    assert summary.count("\n- ") <= 1, "summary reads as a checklist of edits"