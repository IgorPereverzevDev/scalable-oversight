"""Task definition — the only file `inspect eval` needs to see.

Everything the prototype spent code on (run loops, checkpointing to Drive, resume
logic, side-by-side comparison, the capability sweep, the GPU add-on) is now
either handled by the framework or expressed as a command-line flag:

    # Phase 1 baseline: broken v0.1.1 test harness, weak monitor
    inspect eval cerberus/evals/task.py \\
        -T legacy_quoting=true \\
        --model anthropic/claude-sonnet-4-5 \\
        --model-role monitor=openrouter/meta-llama/llama-3.1-8b-instruct

    # Phase 2: fixed harness, same everything else -> measures the fix
    inspect eval cerberus/evals/task.py \\
        --model anthropic/claude-sonnet-4-5 \\
        --model-role monitor=openrouter/meta-llama/llama-3.1-8b-instruct

    # Capability sweep: the agent does NOT run again
    inspect score logs/<run>.eval --model-role monitor=anthropic/claude-haiku-4-5
    inspect score logs/<run>.eval --model-role monitor=openrouter/qwen/qwen3-32b
    inspect score logs/<run>.eval --model-role monitor=vllm/Qwen/Qwen3-8B
"""

import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.util import SandboxEnvironmentSpec

# `inspect eval` loads this file as a standalone module by path and changes the
# working directory to the task folder, so neither relative imports nor an
# implicit "." on sys.path can find the package. Installing it (pip install -e .)
# is the intended setup; this bootstrap keeps a bare checkout working too.
if "cerberus" not in sys.modules:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

# Absolute imports, not relative: under the loader above there is no parent
# package, so `from .agent import ...` raises ModuleNotFoundError. Every other
# module in the package keeps relative imports; this one is the entry point.
from cerberus.evals.agent import MAX_TURNS, cerberus_agent
from cerberus.evals.claude_code import claude_code_agent
from cerberus.evals.dataset import cerberus_dataset
from cerberus.evals.scorers import ground_truth, monitor


_SANDBOX_DIR = Path(__file__).resolve().parents[2] / "sandbox"


def _sandbox_spec(sandbox: str):
    """Resolve the sandbox name to a spec.

    Docker needs a compose file; local needs nothing. Keeping the mapping here
    means the caller says "docker" rather than passing a path, and the compose
    file cannot drift out of sync with the task.
    """
    if sandbox == "docker":
        return SandboxEnvironmentSpec("docker", str(_SANDBOX_DIR / "compose.yaml"))
    if sandbox == "docker_claude_code":
        return SandboxEnvironmentSpec(
            "docker", str(_SANDBOX_DIR / "compose.claude_code.yaml")
        )
    return sandbox


def _solver_for(scaffold: str, max_turns: int):
    if scaffold == "cerberus":
        return cerberus_agent(max_turns=max_turns)
    if scaffold == "claude_code":
        # No turn cap: removing it is the point of the phase.
        return claude_code_agent()
    raise ValueError(f"unknown scaffold {scaffold!r}; use 'cerberus' or 'claude_code'")


@task
def cerberus(
    legacy_quoting: bool = False,
    v011_only: bool = False,
    realistic: bool = False,
    features: bool = False,
    max_turns: int = MAX_TURNS,
    sandbox: str = "local",
    scaffold: str = "cerberus",
) -> Task:
    """Weak-monitor-vs-strong-agent evaluation.

    Args:
        legacy_quoting: Ship the broken v0.1.1 test files. Only for reproducing
            the frozen baseline; see `dataset.build_test_file`.
        v011_only: Restrict to the original 20 tasks. Required when comparing
            against the frozen baseline — a 139-task run and a 20-task run share
            no denominator.
        features: Use the feature-request task set: three multi-file change
            requests in all five categories, 15 samples. Episodes run 26-56 turns
            against 13-16 for single-function tasks, which is the first setting
            in this project where misbehaviour has somewhere to hide. Requires
            scaffold="claude_code" to be worth running.
        realistic: Embed each task in a working project (phase 2g) instead of a
            two-file folder. The task is identical; what changes is how much
            ordinary work surrounds it, and therefore how long the trajectory
            gets. Numbers are not comparable to flat-layout runs — that
            incomparability is the measurement.
        max_turns: Hard cap on agent turns, as in the prototype.
        scaffold: Which agent is under test. "cerberus" is the four-tool loop
            every phase up to 2g used. "claude_code" runs the real CLI through
            Inspect's sandbox bridge (phase 2h) and requires
            sandbox="docker_claude_code" for the Node runtime.

            Numbers from the two scaffolds are not comparable — different tools,
            different trajectory lengths, different token cost. That
            incomparability is the measurement: it is the first setting in this
            project where an episode is long enough for misbehaviour to have
            somewhere to hide.
        sandbox: "local" mirrors the prototype's subprocess isolation and is what
            every number up to phase 2e was measured under. "docker" uses
            sandbox/compose.yaml: real isolation, no network, resource limits.

            Switching is phase 2f and has its own gate — the execution
            environment can move ground-truth labels on its own (different
            Python build, different cwd, different limits), so the first docker
            run keeps the same tasks and the same tools and asks only one
            question: did any label change?
    """
    return Task(
        dataset=cerberus_dataset(
            legacy_quoting=legacy_quoting,
            v011_only=v011_only,
            realistic=realistic,
            features=features,
        ),
        solver=_solver_for(scaffold, max_turns),
        scorer=[ground_truth(), monitor()],
        sandbox=_sandbox_spec(sandbox),
        metadata={
            "environment": "v0.1.1-legacy" if legacy_quoting else "v0.1.2-fixed",
            "task_set": "v0.1.1" if v011_only else "v0.2",
            "layout": "features" if features else ("repo" if realistic else "flat"),
            "scaffold": scaffold,
        },
    )