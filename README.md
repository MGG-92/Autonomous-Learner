# Autonomous Learner

> **Work in progress · Agentic AI experiment**
>
> Most of the implementation, testing, debugging, and documentation in this project
> is done by agentic AI under human direction. Treat this project as an experiment,
> nothing less and nothing more. It is not a finished product, and its results do
> not establish general intelligence or independent language understanding.

A research prototype that learns synthetic facts through teacher interaction and
trains a small PyTorch controller to choose useful questions. The first experiment
separates knowledge acquisition, parameter learning, supplied inference, and language
generation. No pretrained model or paid API is needed for the controlled track.

**Status:** under active development. See [PROJECT_STATE.md](PROJECT_STATE.md) for
verified results, blockers, and the exact next task. See [docs/PLAN.md](docs/PLAN.md)
for the full workflow. Broader memory management and open-ended language learning
remain research milestones; current learned retention uses a narrow task setting.

## Setup

Python 3.11+ is required. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest -q
```

If `python` is not on PATH in Codex desktop, its bundled runtime can bootstrap:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
```

On Linux/macOS substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`.
The bundled runtime is optional, not a project dependency.

## Observe acquisition, restart, transfer, and ablation

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner inspect runs/demo.sqlite
```

The demo begins with UNKNOWN, asks a synthetic teacher, saves evidence, reopens the
database, and answers both the original goal and a new related query. Removing memory
returns UNKNOWN. By default this illustrates a **heuristic** controller. To observe
the trained controller, pass `--checkpoint runs/experiment/controller.pt`.
Use a fresh database path for each demo; it refuses to overwrite previous evidence.

## Train and compare

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner experiment --output runs/experiment
.\.venv\Scripts\python.exe -m autonomous_learner evaluate runs/experiment/controller.pt --start 300000 --worlds 100
```

For a quick experiment, add `--config configs/smoke.json`. JSON configuration fields
are documented by `TrainConfig` in `training.py`. Each output directory contains:

- `controller.pt` and `initial.pt`: learned and untrained parameter checkpoints.
- `curve.csv`: validation learning curve across imitation and policy-gradient training.
- `report.json`: configuration, runtime versions, Git revision/dirty flag, metrics,
  baseline comparisons, and scientific limitations.

Checkpoints are selected using validation reward, never test accuracy. Training,
validation, and final test seeds are checked for overlap. Final metrics include
accuracy, coverage, selective accuracy, teacher calls, steps, reward, elapsed time,
and a Wilson interval across generated worlds. One training seed is not evidence of
robustness across optimizer initializations; use the replication runner below.

For a frozen preliminary replication over three optimizer seeds and a fresh held-out
world range:

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner replicate --config configs/confirmatory.json --seeds 7 42 97 --output runs/replication
```

## What actually learns?

An additional memory experiment trains retention priorities under a hard budget:

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner memory-experiment --output runs/memory
```

It has separate persistent episodic and belief graphs (4 and 3 evidence slots).
After observing a proof, it must retain that proof through distractors. The future
query is provided as a task hint. Consolidation occurs after each observation;
the scorer learns which facts to retain, not when to consolidate. This experiment
is evaluated separately from the question-selection benchmark. Both frozen controllers
can be combined in the tick loop; joint optimization is still pending.

Experience changes the facts in bounded SQLite memory. Optimization changes the
neural action scorer through imitation followed by REINFORCE. The supplied inference
procedure follows `is_a` edges and inherits explicit positive/negative capabilities.
The question-selection benchmark uses FIFO; the separate memory benchmark learns
retention priorities. Output is structured text. These are
deliberate experimental boundaries, not learned grammar or learned inference rules.

The teacher's hidden facts never enter policy features. Expert training labels may
inspect the hidden world; evaluation actions cannot. No-memory evaluation keeps the
same controller, task catalog, and compute budget while denying accessible evidence.

## Bounded autonomous investigation

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/session.json --checkpoint runs/experiment/controller.pt --ticks 2
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/session.json --checkpoint runs/experiment/controller.pt --ticks 8
```

The second invocation resumes the first. The loop persists question history and
budgets, uses application timestamps, and stops after answering or exhausting its
resources. Audit traces are separate from accessible memory. Use one writer per
session. Atomic session-file replacement protects against partially written JSON;
the session and evidence database are not one crash-atomic transaction.
Add `--retention-checkpoint runs/memory/retention.pt` to run the question controller
against learned, bounded dual memory. The two controllers are trained separately;
this option does not imply joint optimization.
Add `--teacher human` for restricted JSON teacher replies. `--teacher local` loads
the optional language model as a separate teacher invocation; its claims are marked
unverified. Neither interface replaces the synthetic oracle in controlled evaluation.

## Optional language and token-control track

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[language]'
.\.venv\Scripts\python.exe -m autonomous_learner language-smoke --output runs/language.json
```

This explicitly downloads the default model, `HuggingFaceTB/SmolLM2-135M-Instruct`,
into ignored `runs/models/`. Use `--model` to change it, `--revision` to pin its source
revision, and `--local-only` for an already-cached offline run. Model weights are
frozen and remote model code is disabled. Candidate probabilities remain normalized
over the full vocabulary. Additional pages reuse the same cached distribution.

The separate token scorer combines the language prior with a trainable residual
and MORE/END actions. A supervised-loss interface supports visible tokens, page
requests, and termination. The generation driver keeps earlier candidates available
after requesting another page. Unit tests demonstrate optimization on a small
synthetic selection task; the real-model smoke uses an **untrained** scorer. No
benefit to language quality or task performance has been established. Model context
is currently recomputed after each emitted token; KV-cache optimization is pending.

Answer rendering copies exact entity names after generating a placeholder sentence.
A strict acceptance check falls back to the template if the proposed sentence
changes. This narrow baseline preserves the claim but does not validate arbitrary
paraphrases or demonstrate learned grammar.

Model/interface references: [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[Transformers outputs](https://huggingface.co/docs/transformers/main_classes/output).

Export an evidence graph with `autonomous-learner graph runs/demo.sqlite --output runs/memory.dot`.
Render the DOT file using Graphviz if installed; inspection does not modify memory.

## Continuing development in another chat

Open this project and say **continue**. Root [AGENTS.md](AGENTS.md) instructs the next
agent to read [PROJECT_STATE.md](PROJECT_STATE.md), run checks, and take the next
unfinished milestone. Git commits and saved experiment reports provide durable
context. A new chat must actually open this checkout; conversation memory alone is
not the continuation mechanism. Development resumes when a user starts a turn;
this repository does not independently restart Codex when usage limits reset.

Never commit virtual environments, private conversations, model caches, credentials,
raw databases, or large checkpoints. Generated artifacts live under ignored `runs/`.
Before pushing, run `python scripts/publication_check.py` and review staged changes.
The public repository publishes code and sanitized synthetic reports only.
