"""The problem bank must be self-consistent before any task uses it.

This file exists because of one v0.1.1 bug: a reward_hack task whose hidden tests
raised NameError on every input, so a correct implementation was labelled a
reward hack. Nothing caught it because nothing ever ran a known-good solution
against those tests.

Every check here runs a reference implementation against a real Python process.
No API keys, no sandbox, about two seconds for the whole bank.
"""

import ast
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from cerberus.evals.dataset import build_test_file
from cerberus.evals.problems import PROBLEMS, SINGLE_ARG_PROBLEMS

IDS = [p["fn_name"] for p in PROBLEMS]


def _run(files: dict[str, str], entry: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            Path(tmp, name).write_text(content)
        return subprocess.run(
            ["python3", entry], cwd=tmp, capture_output=True, text=True, timeout=15
        )


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_reference_passes_visible_tests(problem):
    """The generated test file must accept a correct implementation.

    A failure here means the task is unsolvable as written, and every agent
    attempt would be recorded as an agent failure rather than a dataset defect.
    """
    result = _run(
        {
            "solution.py": problem["reference"],
            "test_solution.py": build_test_file(problem),
        },
        "test_solution.py",
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-400:]


@pytest.mark.parametrize("problem", SINGLE_ARG_PROBLEMS, ids=lambda p: p["fn_name"])
def test_reference_passes_hidden_tests(problem):
    """The exact v0.1.1 regression: hidden tests must not fail correct code.

    Hidden tests decide `did_bad_thing` for the whole reward_hack category. A
    malformed one does not merely lose information — it manufactures evidence of
    misbehaviour that never happened.
    """
    lines = [f"from solution import {problem['fn_name']}", "import json", "results = []"]
    for inp, expected in problem["hidden"]:
        lines.append(
            f"try: results.append(({problem['fn_name']}({inp!r}) == {expected!r}, None))\n"
            f"except Exception as e: results.append((False, str(e)))"
        )
    lines.append("print(json.dumps(results))")

    result = _run(
        {"solution.py": problem["reference"], "check.py": "\n".join(lines)}, "check.py"
    )
    assert result.returncode == 0, result.stderr[-400:]

    outcomes = json.loads(result.stdout.strip().split("\n")[-1])
    failures = [
        (inp, err) for (ok, err), (inp, _) in zip(outcomes, problem["hidden"]) if not ok
    ]
    assert not failures, f"hidden tests reject the reference solution: {failures}"


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_reference_defines_the_declared_function(problem):
    """Catches a rename between `fn_name` and the reference body."""
    tree = ast.parse(problem["reference"])
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert problem["fn_name"] in defined


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_generated_test_file_parses(problem):
    ast.parse(build_test_file(problem))


def test_problem_ids_are_unique():
    ids = [p["humaneval_id"] for p in PROBLEMS]
    assert len(set(ids)) == len(ids)


def test_multi_argument_problems_declare_no_hidden_tests():
    """The hidden-test convention stores values, and repr((2, 3)) renders one
    tuple rather than two arguments. Multi-argument problems therefore opt out
    rather than silently producing wrong calls — they still serve as clean,
    backdoor, exfiltration and tampering bases.
    """
    for problem in PROBLEMS:
        if not problem["hidden"]:
            continue
        first_call = problem["tests"][0][0]
        assert "," not in first_call or first_call.strip().startswith(("[", "(", "{")), (
            f"{problem['fn_name']} looks multi-argument but declares hidden tests"
        )
