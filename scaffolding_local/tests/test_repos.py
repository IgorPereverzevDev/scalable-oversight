"""The realistic workspace must be a working project, not a plausible-looking one.

Every check here runs real Python. A repository that merely looks like a project
would make every task unsolvable for reasons that have nothing to do with the
experiment, and the failure would show up as "the agent got worse".
"""

import ast
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from cerberus.evals.dataset import build_specs, build_test_file
from cerberus.evals.problems import PROBLEMS_BY_ID
from cerberus.evals.repos import TEMPLATES, build_workspace, template_for

REPO_SPECS = build_specs(realistic=True)
FLAT_SPECS = build_specs()


def _materialise(files: dict[str, str], tmp: str) -> None:
    for path, content in files.items():
        target = Path(tmp, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _run(tmp: str, script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", script],
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
    )


# ---------------------------------------------------------------------------
# The templates themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", TEMPLATES.values(), ids=list(TEMPLATES))
def test_template_python_files_parse(template):
    for path, content in template.files.items():
        if path.endswith(".py"):
            ast.parse(content)


@pytest.mark.parametrize("template", TEMPLATES.values(), ids=list(TEMPLATES))
def test_template_own_tests_pass(template):
    """The project's pre-existing tests must be green before the agent arrives.

    A repository shipped with failing tests would invite the agent to "fix" them,
    which is exactly the behaviour one of our categories measures — the dataset
    would be provoking it by accident.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _materialise(template.files, tmp)
        for path in template.files:
            if path.startswith("tests/"):
                result = _run(tmp, path)
                assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-400:]


@pytest.mark.parametrize("template", TEMPLATES.values(), ids=list(TEMPLATES))
def test_template_has_no_todo_markers(template):
    """Deliberately boring: mess is a second variable.

    If detection drops, "the trajectory got longer" and "the monitor got
    distracted by clutter" must not both be available as explanations.
    """
    for content in template.files.values():
        assert "TODO" not in content
        assert "FIXME" not in content


# ---------------------------------------------------------------------------
# Tasks embedded in a project
# ---------------------------------------------------------------------------


def test_repo_layout_produces_a_real_project():
    spec = next(s for s in REPO_SPECS if s["task_id"].startswith("clean_he035"))
    assert len(spec["files"]) >= 8
    assert spec["solution_path"] == "payments/validation.py"
    assert spec["test_path"] == "tests/test_validation.py"
    assert spec["setup"] and "git init" in spec["setup"]


def test_string_problems_land_in_the_string_project():
    """A task transplanted into an unrelated project reads as artificial."""
    assert template_for(PROBLEMS_BY_ID[23]).name == "logparse"   # strlen
    assert template_for(PROBLEMS_BY_ID[35]).name == "payments"   # max_element


@pytest.mark.parametrize(
    "spec",
    [s for s in REPO_SPECS if s["ground_truth_type"] == "clean"][:8],
    ids=lambda s: s["task_id"],
)
def test_reference_solution_passes_inside_the_project(spec):
    """The embedded task must be solvable exactly as the flat one is.

    The stub is replaced by the known-good implementation and the task's test
    file is run from the project root. A failure here means the packaging, not
    the problem, is broken.
    """
    problem = PROBLEMS_BY_ID[spec["humaneval_id"]]
    files = dict(spec["files"])
    files[spec["solution_path"]] = (
        files[spec["solution_path"]].replace(
            f"def {problem['fn_name']}", "def _stub_removed"
        )
        + "\n\n"
        + problem["reference"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        _materialise(files, tmp)
        result = _run(tmp, spec["test_path"])
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-400:]


def test_stub_is_present_and_unimplemented():
    spec = next(s for s in REPO_SPECS if s["task_id"].startswith("clean_he023"))
    module = spec["files"][spec["solution_path"]]
    assert "def strlen" in module
    assert "NotImplementedError" in module


def test_side_objectives_name_real_files():
    """Phrasings are written against the flat layout; in a project those file
    names do not exist, and an instruction naming a missing file is a different
    instruction."""
    for spec in REPO_SPECS:
        if not spec["is_adversarial"]:
            continue
        prompt = spec["user_prompt"]
        assert "test_solution.py" not in prompt, spec["task_id"]
        assert "`solution.py`" not in prompt, spec["task_id"]


def test_layout_paths_reach_ground_truth_params():
    """Without these a checker keeps looking at solution.py and reports
    did_bad_thing=False for every task — plausible numbers, no signal."""
    for spec in REPO_SPECS:
        if spec["ground_truth_type"] == "clean":
            continue
        params = spec["ground_truth_params"]
        assert params["solution_path"] == spec["solution_path"]
        assert params["test_path"] == spec["test_path"]


def test_both_layouts_describe_the_same_tasks():
    """Only the workspace changes. Same ids, same categories, same triggers —
    otherwise a difference in detection could come from the tasks."""
    flat = {s["task_id"]: s for s in FLAT_SPECS}
    repo = {s["task_id"]: s for s in REPO_SPECS}
    assert set(flat) == set(repo)
    for task_id, flat_spec in flat.items():
        repo_spec = repo[task_id]
        assert flat_spec["ground_truth_type"] == repo_spec["ground_truth_type"]
        assert flat_spec["side_objective"] == repo_spec["side_objective"]
        if "trigger_string" in flat_spec["ground_truth_params"]:
            assert (
                flat_spec["ground_truth_params"]["trigger_string"]
                == repo_spec["ground_truth_params"]["trigger_string"]
            )


def test_exfiltration_secret_survives_into_the_project():
    for spec in REPO_SPECS:
        if spec["ground_truth_type"] != "exfiltration":
            continue
        assert spec["ground_truth_params"]["secret_file"] in spec["files"]


def test_flat_layout_is_untouched():
    """The default must remain exactly what phases 0-2f measured."""
    spec = next(s for s in FLAT_SPECS if s["task_id"] == "clean_he023_strlen")
    assert list(spec["files"]) == ["test_solution.py"]
    assert spec["solution_path"] == "solution.py"
    assert spec["setup"] is None
    assert "from solution import strlen" in spec["files"]["test_solution.py"]


# ---------------------------------------------------------------------------
# Regressions from the first realistic smoke run
# ---------------------------------------------------------------------------


def test_every_sample_carries_its_layout_in_metadata():
    """Clean tasks have no ground-truth parameters at all.

    The first realistic run read the layout from `ground_truth_params` and so
    handed clean tasks the flat default. run_tests then looked for a solution.py
    that does not exist, and every clean episode opened with an error the agent
    had to work around. Metadata is the one place present on every sample.
    """
    from cerberus.evals.dataset import cerberus_dataset

    for dataset, expected_root in ((cerberus_dataset(), "solution.py"), ):
        for sample in dataset:
            assert sample.metadata["solution_path"] == expected_root

    for sample in cerberus_dataset(realistic=True):
        solution = sample.metadata["solution_path"]
        assert solution != "solution.py"
        assert solution in sample.files
        assert sample.metadata["test_path"] in sample.files


def test_git_history_is_not_part_of_the_task_files():
    """History is created by the setup script inside the sandbox, not shipped.

    Shipping .git would put hundreds of object files into the workspace snapshot
    and into anything that walks it.
    """
    for spec in REPO_SPECS:
        assert not any(path.startswith(".git/") for path in spec["files"])


def test_solver_publishes_the_layout_into_the_store():
    """The solver must seed the workspace layout before the first tool call.

    Two earlier runs failed here for different reasons — first the value was read
    from the wrong place, then the edit that fixed it silently did not apply.
    Neither was caught, because no test looked at the solver's source and the
    unit tests never execute it.

    Checking the source is crude, but it fails on exactly the class of mistake
    that produced two wasted runs: a solver that quietly leaves the store empty.
    """
    import inspect as pyinspect

    from cerberus.evals import agent

    source = pyinspect.getsource(agent)
    assert "STORE_SOLUTION_PATH" in source
    assert "STORE_TEST_PATH" in source
    assert 'metadata.get("solution_path")' in source
    # The comment in the solver mentions ground_truth_params by name, so match
    # the call rather than the word.
    assert 'ground_truth_params", {}' not in source, (
        "layout must come from sample metadata: clean tasks have no "
        "ground_truth_params, so reading from there hands them the flat default"
    )


@pytest.mark.parametrize(
    "spec",
    [s for s in REPO_SPECS if s["ground_truth_type"] == "test_tampering"][:6],
    ids=lambda s: s["task_id"],
)
def test_pristine_rerun_works_inside_a_project(spec):
    """`task_succeeded` for tampering tasks comes from re-running the ORIGINAL
    tests against the agent's final module.

    The first realistic run reported 0/25 success for this category while the
    agent's own `run_tests` said TESTS PASSED on every one of them. The re-run
    was reading `solution.py` — the flat-layout literal — so it always got an
    empty module, and it executed the test file without the package around it,
    so the import would have failed anyway.
    """
    from cerberus.core.checkers import check
    from cerberus.core.trajectory import TrajectoryView

    problem = PROBLEMS_BY_ID[spec["humaneval_id"]]
    files = dict(spec["files"])
    files[spec["solution_path"]] = (
        files[spec["solution_path"]] + "\n\n" + problem["reference"]
    )
    traj = TrajectoryView(
        task_id=spec["task_id"], messages=[], final_output="", final_files=files
    )
    result = check("test_tampering", spec["ground_truth_params"], traj)

    assert result.did_bad_thing is False
    assert result.task_succeeded is True, result.evidence


def test_tampering_params_carry_the_pristine_project():
    """Without the package the pristine test file cannot even import."""
    for spec in REPO_SPECS:
        if spec["ground_truth_type"] != "test_tampering":
            continue
        pristine = spec["ground_truth_params"]["pristine_files"]
        assert spec["solution_path"] in pristine
        assert spec["test_path"] in pristine