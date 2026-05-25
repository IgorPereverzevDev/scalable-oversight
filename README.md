# Cerberus — Weak Monitor Capability Sweep (v0.1.1)

Experimental infrastructure for studying **scalable oversight**: can a weaker (smaller) model reliably act as a monitor over a stronger coding agent?

The experiment runs one strong agent (Claude Sonnet 4.5) through 20 programming tasks — 10 clean and 10 adversarial — and measures how well eight different monitor models catch the agent's adversarial behavior.

## What's inside

```
.
├── README.md                            # this file
├── weak_monitor_prototype_v0_1_1.ipynb  # main notebook (45 cells)
├── report_v0_1_1.md                     # technical report with results
├── requirements.txt                     # dependencies
├── .gitignore
└── outputs/                             # experiment artifacts
    ├── capability_sweep.json            # MAIN result: 8 monitors × 20 episodes
    ├── trajectories.json                # 20 agent trajectories
    ├── ground_truths.json               # ground truth labels
    ├── sweep_metrics.json               # computed per-monitor metrics
    ├── sweep_verdicts_*.json            # raw verdicts per monitor
    └── qwen_abliterated_verdicts.json   # Qwen abliterated verdicts (self-hosted)
```

## Architecture

Three roles:

- **Strong agent** — Claude Sonnet 4.5, solves tasks via the Anthropic API with tools (write_file, read_file, run_python, run_tests).
- **Weak monitor** — eight observer models, each reads the agent's trajectory and returns a verdict (safe / unsafe / parse_error) with a reason and confidence.
- **Ground truth** — scripted Python checks that deterministically decide whether real adversarial behavior occurred.

All components communicate through JSON contracts with Pydantic schemas. Every object carries a `schema_version` for backward compatibility.

## Adversarial task categories

- **Backdoor (4)** — a hidden trigger disguised as a "documented feature" (`__ADMIN_OVERRIDE__`, master key, etc.).
- **Reward hack (3)** — fitting the code to the visible tests instead of solving the task honestly.
- **Exfiltration (3)** — reading secret files and embedding their contents in the output.

## The eight monitors

| Monitor | Provider |
|---|---|
| Llama 3.1 8B | OpenRouter |
| Qwen3 8B | OpenRouter |
| Qwen3 32B | OpenRouter |
| Llama 3.3 70B | OpenRouter |
| DeepSeek V3 | OpenRouter |
| DeepSeek R1 | OpenRouter |
| Claude Haiku 4.5 | Anthropic API |
| Qwen3.6 27B abliterated | self-hosted (Colab G4) |

## Key findings

Full details in `report_v0_1_1.md`. In short:

1. **The capability gap is category-bound, not size-bound.** None of the seven standard monitors caught a single backdoor (0 of 28). Exfiltration is caught reliably from mid-size models upward.
2. **The high FPR of strong monitors is test tampering**, an unlabeled category in our ground truth, not false alarms.
3. **Reasoning can hurt detection** — DeepSeek R1 (with reasoning) caught fewer than V3 (without), rationalizing the prompt.
4. **The abliterated model catches backdoors** — Qwen3.6 27B abliterated was the only monitor to catch any backdoor that every safety-trained model missed.

## How to run

The notebook targets **Google Colab** (self-hosting Qwen needs a GPU with ≥60GB VRAM — G4 / A100-80GB / H100).

### Environment

```bash
pip install -r requirements.txt
```

For Qwen abliterated, additionally (GPU phase only):

```bash
pip install --upgrade "transformers>=5.5,<5.6" accelerate safetensors
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention
# restart the runtime after installing the kernels
```

### API keys

The notebook reads keys from Colab Secrets (`userdata.get`):

- `OPENROUTER_API_KEY` — for the six monitors via OpenRouter
- `ANTHROPIC_API_KEY` — for the agent and Haiku

### Run phases

- **Phase 1 (CPU runtime):** Sections 1-14 — agent run and the sweep over the seven API monitors.
- **Phase 3 (GPU runtime):** Section 15 — self-hosting Qwen abliterated. Requires switching to GPU and restarting.

The notebook is **resumable** — sections load cached results from disk if they exist. Artifacts are saved to Google Drive so they survive a runtime switch.

## Self-hosting Qwen3.6 27B abliterated — critical

The model uses a hybrid GatedDeltaNet architecture (48 GDN + 16 attention layers). The working configuration was found after seven failed attempts. Requirements:

- `transformers >= 5.5` (for `qwen3_5` support)
- `causal-conv1d` (install with `--no-build-isolation`)
- `flash-linear-attention`
- **the whole model in BF16 on the GPU, no CPU offload** (`device_map="cuda:0"`, no `max_memory`)
- restart the runtime after installing the kernels

CPU offload or int8 quantization produce broken output. See the report for details.

## Limitations

20 episodes is a prototype, not a basis for strong conclusions. Reward hack barely accumulated (Sonnet refused 2 of 3). Single agent (Sonnet). Qwen has 5/20 parse_errors. More in the report's limitations section.

## Roadmap (v0.2)

- Expand the dataset to 100+ tasks, add test tampering as an explicit category
- Multiple agents (Opus, GPT, Gemini) to test generalization
- AI-based auditor instead of scripted prompts (Petri-style)
- Anti-rationalization prompt experiments for reasoning monitors

## Version

v0.1.1. Schema version is recorded in every artifact.
