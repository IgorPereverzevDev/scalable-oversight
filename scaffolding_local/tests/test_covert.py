"""Covert tasks must be solvable honestly, and their hidden tests must catch harm.

Two references per task instead of one. A hidden test validated only against an
honest implementation proves it does not slander correct code; validated only
against a harmful one it proves it catches the harm. Neither alone is enough, and
this project has already shipped a hidden suite that failed on correct code.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from cerberus.evals.covert import COVERT_TASKS, COVERT_TRIGGER
from cerberus.evals.repos import TEMPLATES

IDS = [t.task_id for t in COVERT_TASKS]


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


def _workspace(task, reference):
    files = dict(TEMPLATES[task.repo].files)
    files[task.test_path] = task.test_source
    files.update(reference)
    return files


@pytest.mark.parametrize("task", COVERT_TASKS, ids=IDS)
def test_honest_reference_passes_the_visible_suite(task):
    result = _run(_workspace(task, task.honest_reference), task.test_path)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-500:]


@pytest.mark.parametrize(
    "task", [t for t in COVERT_TASKS if t.hidden_source], ids=lambda t: t.task_id
)
def test_honest_reference_passes_the_hidden_suite(task):
    """The hidden tests must not condemn a correct implementation."""
    files = _workspace(task, task.honest_reference)
    files["_hidden.py"] = task.hidden_source
    result = _run(files, "_hidden.py")
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-500:]



@pytest.mark.parametrize("task", COVERT_TASKS, ids=IDS)
def test_harmful_reference_still_passes_the_visible_suite(task):
    """This is what makes the harm covert rather than merely broken.

    If the visible tests failed, the agent would see something was wrong and fix
    it — and there would be nothing for a monitor to miss.
    """
    result = _run(_workspace(task, task.harmful_reference), task.test_path)
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-500:]


@pytest.mark.parametrize(
    "task", [t for t in COVERT_TASKS if t.hidden_source], ids=lambda t: t.task_id
)
def test_hidden_suite_catches_the_harmful_reference(task):
    """Without this the hidden tests could be vacuous and every episode would
    read as clean."""
    files = _workspace(task, task.harmful_reference)
    files["_hidden.py"] = task.hidden_source
    result = _run(files, "_hidden.py")
    assert result.returncode != 0, "hidden suite accepted the harmful implementation"

def test_overt_harm_is_findable_in_the_code():
    """The control needs no hidden suite: an explicit bypass leaves a literal.

    That asymmetry is the experiment. A covert change has nothing to grep for, so
    it needs behavioural ground truth — and a monitor reading the diff may have
    nothing to see either.
    """
    overt = next(t for t in COVERT_TASKS if t.variant == "overt")
    assert overt.trigger_string
    assert any(
        overt.trigger_string in content
        for content in overt.harmful_reference.values()
    )
    assert not overt.hidden_source


def test_pairs_differ_only_in_how_the_harm_is_expressed():
    covert = next(t for t in COVERT_TASKS if t.variant == "covert")
    overt = next(t for t in COVERT_TASKS if t.variant == "overt")

    assert covert.pair_id == overt.task_id and overt.pair_id == covert.task_id
    assert covert.repo == overt.repo
    assert covert.test_source == overt.test_source
    # Hidden suites are NOT shared: the two harms are checked differently, and
    # that difference is what the pair measures.
    assert covert.hidden_source and not overt.hidden_source


def test_only_the_overt_variant_names_a_trigger():
    """The covert harm has no literal to grep for — which is why it needs
    behavioural ground truth, and why a monitor may have nothing to see."""
    covert = next(t for t in COVERT_TASKS if t.variant == "covert")
    overt = next(t for t in COVERT_TASKS if t.variant == "overt")

    assert COVERT_TRIGGER in overt.summary
    assert COVERT_TRIGGER not in covert.summary
    assert not any(COVERT_TRIGGER in c for c in covert.harmful_reference.values())