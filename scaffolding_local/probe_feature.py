"""Одноразовая проба: дают ли feature-задачи длинные траектории.

Собирает временный таск из одной задачи, без adversarial-обвязки и категорий.
Нужен один ответ: сколько сообщений в эпизоде против 13 у обычных задач.
"""
import sys

from inspect_ai import Task, eval as run_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

from cerberus.evals.claude_code import claude_code_agent
from cerberus.evals.features import FEATURE_TASKS_BY_ID, feature_workspace
from cerberus.evals.repos import TEMPLATES, git_setup_script

task_id = sys.argv[1] if len(sys.argv) > 1 else "feat_payments_jpy"
feature = FEATURE_TASKS_BY_ID[task_id]
template = TEMPLATES[feature.repo]

sample = Sample(
    id=feature.task_id,
    input=feature.summary,
    files=feature_workspace(feature, TEMPLATES),
    setup=git_setup_script(template),
    metadata={
        "user_prompt": feature.summary,
        "solution_path": feature.touches[0],
        "test_path": feature.test_path,
    },
)

log = run_eval(
    Task(
        dataset=MemoryDataset([sample]),
        solver=claude_code_agent(),
        sandbox=SandboxEnvironmentSpec("docker", "sandbox/compose.claude_code.yaml"),
    ),
    model="anthropic/claude-sonnet-4-5",
)[0]

s = log.samples[0]
print("\n=== ", feature.task_id)
print("messages:", len(s.messages))
print("stop:", s.store.get("cerberus:stop_reason"), "| rc:", s.store.get("cerberus:cli_returncode"))
print("модифицировано файлов:", sum(
    1 for p, c in (s.store.get("cerberus:final_files") or {}).items()
    if c != feature_workspace(feature, TEMPLATES).get(p)
))