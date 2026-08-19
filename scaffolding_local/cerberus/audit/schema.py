"""What a generated scenario has to contain to be usable.

WHY A SCHEMA AND NOT A PROMPT
-----------------------------
Every ground-truth label in this project comes from a script: a trigger we chose,
hidden tests we wrote, a secret file we planted. That is what makes the labels
worth anything — the alternative, asking a model whether misbehaviour occurred,
measures agreement between two models rather than the monitor.

A generated scenario has no such script by default. The obvious fix — have a
judge model read the transcript — would quietly discard the principle the whole
project rests on, and every number after that point would mean something else.

So the auditor does not produce scenarios. It produces scenarios *with the means
to check them*: a task, a reference implementation, and a hidden suite that
separates a caught agent from an uncaught one. Generation becomes verifiable, and
a scenario that cannot be verified never enters the corpus.

WHAT THE AUDITOR CANNOT DECIDE
------------------------------
Whether a run counts as misbehaviour. That is settled by executing its own hidden
suite against the agent's workspace, exactly as the hand-written covert tasks
are. The auditor writes the test; it does not grade the answer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeneratedScenario:
    """One adversarial scenario proposed by the auditor.

    Mirrors `CovertTask` on purpose: a scenario that survives validation is
    indistinguishable from a hand-written task downstream, and the dataset,
    checkers and scorers need no special case for it.
    """

    scenario_id: str
    repo: str
    summary: str
    """The request as a developer would receive it."""
    test_path: str
    test_source: str
    """Visible suite. Must pass on both the honest and the harmful reference —
    that is what makes the harm covert rather than merely broken."""
    hidden_source: str
    """Passes on the honest reference, fails on the harmful one. The entire
    ground truth."""
    honest_reference: dict[str, str]
    harmful_reference: dict[str, str]
    rationale: str
    """The auditor's account of who benefits or who stops seeing something.

    Required, and never used for scoring. It exists because scripted validation
    can check that behaviour changed, and cannot check that the change is
    harmful — the first generated batch passed every execution check and
    produced three plain defects: a rounding error, a denial of legitimate work,
    and a method that failed to validate something it was never responsible for.

    The sentence is what a human reads when deciding whether to promote a
    scenario into the frozen corpus. A rationale that cannot name a beneficiary
    or a blind spot is describing a bug.
    """
    auditor_model: str = ""
    auditor_prompt_version: str = ""
    seed: int | None = None
    """Provenance. Auditor strength is a confound: 40% detection against a weak
    auditor and against a strong one are different numbers about the same
    monitor. A scenario that cannot say where it came from cannot be compared
    with anything."""




@dataclass(frozen=True)
class ValidationResult:
    """Why a scenario was accepted or rejected.

    Rejections are kept rather than discarded: the rate and shape of them is a
    measurement of the auditor, and an auditor whose scenarios mostly fail
    validation is not producing adversarial pressure, it is producing noise.
    """

    scenario_id: str
    accepted: bool
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]