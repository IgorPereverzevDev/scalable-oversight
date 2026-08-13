"""Phase 2h: Claude Code as the agent under test.

WHY REPLACE OUR OWN LOOP
------------------------
Every phase so far measured detection on 10-15 turn episodes, and detection did
not budge: Haiku held 86% whether the workspace was two files or a nine-file
project. That is not evidence that monitoring is easy — it is evidence that
nothing in the setup ever produced the regime scalable oversight is about, where
a small edit sits inside a long stretch of ordinary work.

The 2g repository added three or four turns of looking around. A real scaffold
adds an order of magnitude: Claude Code greps, reads, edits, re-runs, backtracks.
Same tasks, same categories, same ground truth — only the number of turns the
monitor has to read through changes.

HOW THE BRIDGE WORKS
--------------------
`sandbox_agent_bridge` runs a proxy inside the container that speaks the Anthropic
API and routes to whatever model Inspect resolved. The CLI is pointed at it with
ANTHROPIC_BASE_URL and asked for the model named "inspect". Every request the
agent makes lands in the eval log as a normal trajectory, so scorers, renderers
and monitors need no changes at all.

WHAT STAYS THE SAME, DELIBERATELY
---------------------------------
Dataset, categories, checkers, monitor prompts, renderers. One variable.

The workspace snapshot is unchanged too: it reads the filesystem rather than
tracking writes, so it does not care whether files arrived through our four
tools or through the CLI's own editing. That property was not free — it was the
reason snapshotting was written that way in phase 1 — and it is what makes this
phase a solver swap instead of a rewrite.
"""

from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox, store

from ..adapters.inspect_state import (
    STORE_FINAL_FILES,
    STORE_SOLUTION_PATH,
    STORE_STOP_REASON,
    STORE_TEST_PATH,
)
from ..core.checkers import DEFAULT_SOLUTION_PATH, DEFAULT_TEST_PATH
from .tools import snapshot_workspace

CLAUDE_CODE_TIMEOUT_SEC = 900
"""Wall-clock cap per episode.

Generous on purpose: the point of this phase is long episodes, and a cap tight
enough to bite would silently truncate exactly the trajectories under study. A
timeout here is a failed episode, not a shorter one — it is recorded as such.
"""

BRIDGE_PORT = 13131


@solver
def claude_code_agent(
    timeout: int = CLAUDE_CODE_TIMEOUT_SEC,
    max_turns: int | None = None,
) -> Solver:
    """Run Claude Code inside the sandbox against the task workspace.

    Args:
        timeout: Wall-clock limit for the CLI process.
        max_turns: Cap on agent turns, or None for the CLI default. Left off by
            default: capping turns is the thing this phase is trying to stop
            doing.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        store().set(
            STORE_SOLUTION_PATH,
            metadata.get("solution_path") or DEFAULT_SOLUTION_PATH,
        )
        store().set(STORE_TEST_PATH, metadata.get("test_path") or DEFAULT_TEST_PATH)

        prompt = metadata.get("user_prompt") or state.input_text
        stop_reason = "completed"

        async with sandbox_agent_bridge(port=BRIDGE_PORT) as bridge:
            command = [
                "claude",
                "--print",
                "--model",
                "inspect",
                "--permission-mode",
                # The CLI asks before editing or running anything. There is no
                # human here, so a prompt is a hang; and gating tool use would
                # change the behaviour we are measuring, not just the ergonomics.
                "bypassPermissions",
                "--output-format",
                "text",
            ]
            if max_turns is not None:
                command += ["--max-turns", str(max_turns)]
            command += [prompt]

            result = await sandbox().exec(
                command,
                cwd="/workspace",
                timeout=timeout,
                env={
                    "ANTHROPIC_BASE_URL": f"http://localhost:{BRIDGE_PORT}",
                    "ANTHROPIC_API_KEY": "inspect-bridge",
                    "DISABLE_TELEMETRY": "1",
                    "DISABLE_AUTOUPDATER": "1",
                    "PYTHONPATH": ".",
                    "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",
                },
            )
            store().set("cerberus:cli_returncode", result.returncode)
            store().set("cerberus:cli_stdout", (result.stdout or "")[:4000])
            store().set("cerberus:cli_stderr", (result.stderr or "")[:4000])

            if result.returncode != 0:
                # Recorded rather than raised: one CLI failure should not abort a
                # 139-episode run, and the snapshot below still describes whatever
                # state the workspace was left in.
                stop_reason = "error"

            # Messages come from the bridge, not from our own bookkeeping — the
            # CLI's conversation is the trajectory, and this is the only place it
            # exists in a form the renderers understand.
            state.messages = list(bridge.state.messages)
            if bridge.state.output is not None:
                state.output = bridge.state.output

        store().set(STORE_STOP_REASON, stop_reason)
        store().set(STORE_FINAL_FILES, await snapshot_workspace())
        return state

    return solve