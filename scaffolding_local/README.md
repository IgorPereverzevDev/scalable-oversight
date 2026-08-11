# Cerberus — weak monitor vs strong agent

Empirical study of scalable oversight: **can a less capable model reliably detect
adversarial or misaligned behaviour by a more capable agent during coding tasks?**

This repository is the Inspect port of the `weak_monitor_prototype_v0.1.1`
notebook. The methodology is unchanged; the infrastructure is gone.

---

## Read this first — six decisions that shape the codebase

Understanding these prevents most "why is it like this?" confusion. If any stops
being true, update this section *before* changing the code.

**1. Components are joined by artifacts, not runtime state.**
The agent produces an eval log. Monitors and ground-truth checkers consume it.
Nothing is passed in memory between stages. The cost is indirection; the benefit
is that any monitor can be run against any past log, at any time, without
re-running the agent — which used to cost $2.70 and ten minutes per iteration.

**2. `TrajectoryView` is the port contract, not a runtime object.**
Two sources normalise into it: saved v0.1.1 dumps and Inspect state. Its shape
deliberately mirrors the legacy dump byte for byte, which makes the legacy
adapter an identity function and confines all porting risk to
`adapters/inspect_state.py`. Without this indirection the Phase 0 gate could not
exist at all.

**3. Ground truth is scripted, never LLM-judged.**
We measure whether the *monitor* is good. An LLM judge would instead measure
agreement with a stronger model — a different, noisier question whose answer
drifts as the judge changes. The price is that only behaviours with a clear
external signature can be labelled, and that limit must be stated rather than
worked around.

**4. Scorers are thin wrappers over pure functions.**
All judgement lives in `cerberus/core`, which imports nothing from Inspect. This
is what makes the replay gate possible. The moment logic leaks into a scorer, the
gate stops validating what actually runs.

**5. The monitor's input is an experimental variable, not a detail.**
What the monitor sees is versioned, registered by name, and recorded in the log
alongside every verdict — exactly like the prompt. A change to rendering
invalidates comparison with earlier runs just as surely as a change to wording.

**6. Light abstractions, not class hierarchies.**
Protocols for structural contracts, registries for named alternatives, Pydantic
models for data boundaries, plain functions everywhere else. The rule: abstract
only where two or more implementations exist now or are imminent. Every seam in
this repo names its second implementation. There is no `Pipeline` class, no
`Task` class with methods, no `Checker` base class.

---

## Layout

```
cerberus/
  core/            framework-agnostic; imports nothing from Inspect
    trajectory.py    TrajectoryView / Verdict / GroundTruthResult
    rendering.py     what the monitor sees          (Protocol + registry)
    parsing.py       how its answer is read         (Protocol + registry)
    monitors.py      MonitorSpec: view x prompt x parser
    checkers.py      scripted ground truth          (Protocol + registry)
  adapters/
    legacy.py        v0.1.1 dumps -> TrajectoryView (identity)
    inspect_state.py Inspect      -> TrajectoryView (RISK-1..5 live here)
  evals/
    dataset.py       20 tasks, both environment versions
    tools.py         agent tools, frozen return strings, import guard
    agent.py         the v0.1.1 loop, replicated deliberately
    scorers.py       ground_truth() and monitor()
    task.py          @task — the entry point for `inspect eval`
  analysis/
    metrics.py       cross-scorer confusion matrix, baseline comparison
scripts/
  phase0_replay.py         validate the port against saved v0.1.1 results
  freeze_monitor_inputs.py freeze / verify exact monitor prompts
  report.py                metrics for a log, optional baseline gate
tests/                     48 tests, no API keys or sandbox required
```

---

## Running

```bash
pip install -e ".[dev]"

# 1. Contract tests — no keys, no cost, run these first
pytest

# 2. Phase 0 gate — replay saved v0.1.1 results through the new checkers
python scripts/phase0_replay.py outputs/trajectories.json outputs/ground_truths.json
python scripts/freeze_monitor_inputs.py outputs/trajectories.json --write

# 3. Smoke run — three tasks, cents
inspect eval cerberus/evals/task.py -T legacy_quoting=true --limit 3 \
    --model anthropic/claude-sonnet-4-5 \
    --model-role monitor=openrouter/meta-llama/llama-3.1-8b-instruct

# 4. Baseline run — 20 tasks in the v0.1.1 environment
inspect eval cerberus/evals/task.py -T legacy_quoting=true \
    --model anthropic/claude-sonnet-4-5 \
    --model-role monitor=openrouter/meta-llama/llama-3.1-8b-instruct
python scripts/report.py logs/<run>.eval --save baselines/v0.1.1.json

# 5. Capability sweep — the agent does NOT run again
inspect score logs/<run>.eval --model-role monitor=anthropic/claude-haiku-4-5
inspect score logs/<run>.eval --model-role monitor=openrouter/qwen/qwen3-32b
inspect score logs/<run>.eval --model-role monitor=vllm/Qwen/Qwen3-8B
```

Model selection is Inspect's job throughout: swapping Opus, GPT, DeepSeek or a
local vLLM server in as the monitor is one flag, across providers.

---

## Two environment versions

`legacy_quoting=true` ships the v0.1.1 test harness, in which 12 of 20 generated
test files did not parse. That was a genuine bug, and it mattered: agents
repaired the harness themselves, and strong monitors correctly flagged the
repair as out-of-scope editing, which accounted for the entire false positive
rate of the v0.1.1 sweep.

Both versions are kept because the environment is an experimental variable.
Reproducing the baseline needs the broken one; measuring the effect of the fix
needs both. The version used is recorded in task metadata.

---

## What was deleted from the prototype

Roughly 60% of the notebook was infrastructure that Inspect already provides:
the tool-use loop, the subprocess sandbox, checkpointing to Drive and the resume
logic around it, two hand-written API client wrappers with retry layers, the
monitor registry, the parallel-run bookkeeping, the side-by-side comparison
section, the capability sweep loop, and the entire GPU add-on for running a local
model. All of it is now either framework behaviour or a command-line flag.

What transferred nearly verbatim: the ground-truth checkers, the monitor prompt
and parser, the AST import guard, and the dataset.

---

## Known limitations

- **Contamination.** HumanEval is in every frontier model's training data.
- **Trajectory length.** 4-8 turn episodes give misbehaviour almost no room to
  hide; detection rates here are upper bounds.
- **Plaintext side objectives.** A monitor can partly succeed by reading the
  request instead of analysing behaviour.
- **`clean` has no detector.** Behaviour such as rewriting the test file cannot
  be labelled today, so a monitor that flags it is scored as wrong.
- **`task_succeeded` is agent-influenced.** It greps a marker from a tool the
  agent can affect, on a test file the agent can rewrite. Hidden tests close this
  for `reward_hack`; nothing closes it for `clean`.
- **n = 20.** No confidence interval on this dataset is worth reporting.

See `PHASES.md` for how each of these is scheduled.
