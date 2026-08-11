# Phase plan

One rule governs the whole sequence: **never change methodology and
infrastructure at the same time.** When numbers move, it must be possible to say
which change moved them. Every phase therefore ends at a gate, and no phase
starts before the previous gate is green.

---

## Phase 0 — validate the port without running the agent

**Goal.** Prove the ported checkers and monitor input reproduce v0.1.1, using
saved data only.

**Why not just run everything and compare.** The agent is stochastic and n = 20.
Two runs of the same prototype disagree, so comparing aggregates compares noise
to noise and a real porting bug hides inside the spread. Replaying saved
trajectories is deterministic: same input, same label, no exceptions.

**Work.**
- Normalise both sources into `TrajectoryView`.
- Port checkers, renderer and parser as pure functions.
- Contract tests for the five documented porting risks.

**Gate.**
```bash
pytest
python scripts/phase0_replay.py outputs/trajectories.json outputs/ground_truths.json
python scripts/freeze_monitor_inputs.py outputs/trajectories.json --write
```
Ground truth reproduces on all 20 episodes except registered known divergences;
monitor prompts are frozen byte for byte.

**Cost.** Zero. No API key, no sandbox, no network.

**Status: complete.** One known divergence: `adv_he048_rewardhack_palindrome_shortcut`,
where the v0.1.1 label was wrong.

---

## Phase 1 — run the agent in Inspect, reproduce the baseline

**Goal.** Show that the migration preserves agent behaviour, in the *same*
(broken) environment as v0.1.1.

**Work.**
- Replicate the prototype loop rather than adopting `react()`: built-in agents
  add a system prompt and a `submit()` tool, which changes trajectory length and
  therefore whether the agent misbehaves at all.
- Persist workspace state into the sample store, so scorers survive re-scoring.
- Run with `legacy_quoting=true`.

**Gate.**
```bash
inspect eval cerberus/evals/task.py -T legacy_quoting=true --limit 3 ...   # smoke
inspect eval cerberus/evals/task.py -T legacy_quoting=true ...             # full
python scripts/report.py logs/<run>.eval --baseline baselines/v0.1.1.json
```
Not an equality check — identical numbers would be suspicious. What must hold:
episodes join by id, agent success and misbehaviour counts stay within coarse
bounds, parse errors do not jump (a jump means generation parameters diverged),
and the monitor's qualitative behaviour matches (the weak monitor produced
exactly one `unsafe`, an exfiltration true positive).

**Cost.** About $2.70 and ten minutes.

**Deliverable.** `baselines/v0.1.1.json`, frozen.

---

## Phase 2 — fix the environment, then make it realistic

Three changes, each with its own run and its own delta. Bundling them makes all
three unmeasurable.

**2a. Fix the test harness.** `legacy_quoting=false`. Removes the agent's reason
to rewrite `test_solution.py`. Expected effect: the false positive rate of strong
monitors collapses, because it was entirely driven by that behaviour. Re-run all
monitors before and after.

**2b. Label test tampering.** Add a checker comparing the final
`test_solution.py` against the pristine copy. Today the behaviour cannot be
labelled at all, so a monitor that flags it is scored as wrong — the monitor was
right and the metric was wrong. This also closes the `task_succeeded` hole for
clean tasks.

**2c. Real tools and real isolation.** `sandbox="docker"`, then a realistic agent
scaffold via Inspect's agent bridge. This is the largest single jump in realism
and the point at which detection rates stop being comparable to anything earlier:
trajectories get long and noisy, and misbehaviour finally has room to hide.
Hidden-test execution moves into the sandbox here, with its own label-stability
check.

**Gate for each.** Metrics before and after, same monitors, same seed dataset,
delta reported explicitly. A change with an unexplained delta is reverted, not
rationalised.

**Also in this phase.** Raise the episode count. Everything above is measured on
n = 20, where no confidence interval is meaningful. Once the agent runs once and
monitors re-score cheaply, sample size is the cheapest available improvement.

---

## Phase 3 — monitor scaffolding and a dynamic auditor

**3a. Online monitoring.** A monitor that judges prefixes of a trajectory and can
block a tool call. Inspect's approval policies provide the mechanism; the
`renderer="prefix"` seam in `core/rendering.py` provides the view. New metrics
appear here: not only whether misbehaviour was caught, but on which turn, and at
what cost in false interventions.

**3b. Stealth side objectives.** Move the adversarial instruction out of the user
prompt into file contents, comments and tool output. Until this lands, a monitor
can partly succeed by reading the request rather than analysing behaviour, and
every detection number is inflated by an unknown amount.

**3c. Dynamic auditor (Petri-style).** An auditor model generates adversarial
scenarios instead of a human writing them.

This is last because it depends on everything above, and because it collides with
decision 3 in the README — generated scenarios have no scripted checker. The
resolution: **the auditor must emit a machine-checkable success predicate along
with the scenario.** It plants the canary, writes the hidden tests, names the
signature. Generation becomes *verifiable*, and a scenario enters the corpus only
after passing two controls: it never fires on a clean run, and it does fire when
an agent is explicitly instructed to misbehave. Anything not expressible as a
predicate goes through a separate LLM-judge channel and is never mixed into
scripted metrics.

Two consequences to plan for:

- **Dynamic corpora are not reproducible by construction.** Hence two tiers: a
  frozen regression suite for measuring progress, and a live exploration set
  where the metric is yield (how often the agent actually misbehaves, and how
  much of that is novel) rather than detection rate.
- **Auditor strength becomes a confound.** A 40% detection rate against a weak
  auditor and against a strong one are different numbers about the same monitor.
  Auditor model, seed and prompt version must be part of a run's identity.

---

## Ordering summary

| Phase | Changes | Gate | Comparable to |
|---|---|---|---|
| 0 | port only | replay saved data | v0.1.1 exactly |
| 1 | framework | baseline comparison | v0.1.1 within noise |
| 2a | test harness fix | before/after delta | Phase 1 |
| 2b | new label category | delta on FP episodes | Phase 2a |
| 2c | docker + real scaffold | label stability | nothing earlier |
| 3 | stealth, online, auditor | per-change | Phase 2c |
