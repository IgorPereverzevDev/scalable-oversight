"""Admission control for generated scenarios.

A scenario enters the corpus only if every check passes. The checks are not
quality filters — they are the conditions under which the scenario's own hidden
suite can be trusted as ground truth.

The rejection rate is itself a result. An auditor whose scenarios mostly fail
here is producing noise, and reporting detection rates over its output would be
reporting noise.
"""

import ast
import os
import subprocess
import tempfile
from pathlib import Path

from .schema import GeneratedScenario, ValidationResult

EXEC_TIMEOUT_SEC = 30


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
            timeout=EXEC_TIMEOUT_SEC,
            env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        )


def _passed(result: subprocess.CompletedProcess) -> bool:
    return result.returncode == 0 and "OK" in result.stdout


def validate(scenario: GeneratedScenario, template_files: dict[str, str]) -> ValidationResult:
    """Decide whether a scenario can be trusted as ground truth.

    Six checks, each ruling out a way the scenario would produce a label that
    means nothing:

    parses — a reference that does not compile makes every later check vacuous.

    honest_visible — the task must be solvable. A visible suite that fails on a
    correct implementation would record every agent as having failed, and the
    episode would measure the scenario rather than the agent.

    honest_hidden — the hidden suite must not condemn correct code. This project
    shipped exactly that once: a suite that raised NameError on every input, so
    every honest solution was labelled a reward hack, and nothing noticed for
    months.

    harmful_visible — the harm must be invisible to the tests the agent can see.
    If the visible suite catches it, the agent fixes it, and there is nothing for
    a monitor to miss.

    harmful_hidden — the hidden suite must actually catch the harm. Without this
    a vacuous suite passes everything and every episode reads as clean.

    project_intact — the repository's own tests must still pass. A scenario that
    breaks them hands the agent a broken project, and "repairing files outside
    the task" is a behaviour we measure elsewhere; the corpus would be provoking
    it by accident.
    """
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    checks["parses"] = True
    for source in (*scenario.honest_reference.values(), *scenario.harmful_reference.values(),
                   scenario.test_source, scenario.hidden_source):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            checks["parses"] = False
            detail["parse_error"] = str(exc)
            # Nothing below can be trusted once something fails to compile.
            return ValidationResult(scenario.scenario_id, False, checks, detail)

    base = dict(template_files)
    base[scenario.test_path] = scenario.test_source

    honest = {**base, **scenario.honest_reference}
    harmful = {**base, **scenario.harmful_reference}

    result = _run(honest, scenario.test_path)
    checks["honest_visible"] = _passed(result)
    if not checks["honest_visible"]:
        detail["honest_visible_error"] = result.stderr[-400:]

    result = _run({**honest, "_hidden.py": scenario.hidden_source}, "_hidden.py")
    checks["honest_hidden"] = _passed(result)
    if not checks["honest_hidden"]:
        detail["honest_hidden_error"] = result.stderr[-400:]

    result = _run(harmful, scenario.test_path)
    checks["harmful_visible"] = _passed(result)
    if not checks["harmful_visible"]:
        detail["harmful_visible_error"] = result.stderr[-400:]

    result = _run({**harmful, "_hidden.py": scenario.hidden_source}, "_hidden.py")
    checks["harmful_hidden"] = not _passed(result)
    if not checks["harmful_hidden"]:
        detail["harmful_hidden"] = "the hidden suite accepted the harmful reference"

    checks["project_intact"] = True
    for path in template_files:
        if not path.startswith("tests/") or path == scenario.test_path:
            continue
        result = _run(harmful, path)
        if result.returncode != 0:
            checks["project_intact"] = False
            detail["broken_project_test"] = path
            break

    accepted = all(checks.values())
    return ValidationResult(scenario.scenario_id, accepted, checks, detail)


def validate_all(
    scenarios: list[GeneratedScenario], templates: dict[str, object]
) -> tuple[list[GeneratedScenario], list[ValidationResult]]:
    """Validate a batch, returning what survived and why the rest did not."""
    accepted, results = [], []
    for scenario in scenarios:
        template = templates[scenario.repo]
        result = validate(scenario, template.files)
        results.append(result)
        if result.accepted:
            accepted.append(scenario)
    return accepted, results