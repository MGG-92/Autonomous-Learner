# Autonomous Learner

> **Work in progress · Agentic AI experiment**
>
> Most of the implementation, testing, debugging, and documentation in this project
> is done by agentic AI under human direction. Treat this project as an experiment,
> nothing less and nothing more. It is not a finished product, and its results do
> not establish general intelligence or independent language understanding.

A research prototype investigating a cognitive core that learns internal models
through experience, chooses experiments, and decides what to communicate before
optional language rendering. The first rule-discovery baseline infers unfamiliar
finite operations from examples without an LLM. Its hypothesis language and search
algorithm are supplied; it has not learned general reasoning or invented concepts.
Earlier PyTorch experiments learn question selection and memory control around
programmed graph inference. No paid API is needed for the controlled track.

The long-term objective is learning about worlds with objects, properties, actions,
and consequences, then using learned models to pursue goals and transfer knowledge.
Mathematical rule discovery is an initial diagnostic. Action-world learning,
reusable abstraction discovery, and transfer across world mechanisms remain untested.

**Status:** under active development. See [PROJECT_STATE.md](PROJECT_STATE.md) for
verified results, blockers, and the exact next task. See [docs/PLAN.md](docs/PLAN.md)
for the full workflow. The current priority is internal rule/model learning (C1–C4),
with token control, further memory infrastructure, and viewer polish deferred.

Preliminary results and their limits are in [reports/README.md](reports/README.md).

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

For the active unfamiliar-rule experiment:

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery-smoke.json --output runs/rules-smoke
```

The learner updates competing executable hypotheses from observations and selects
queries with a supplied information-gain heuristic. It predicts five reserved inputs
shared across policies and compares random, fixed/untrained, no-memory, and model-reset
controls. Output includes an actual query trace and structured prediction messages.
No language weights, embeddings, or generated hypotheses are used. Learning here is
belief updating; no neural policy or reasoning algorithm is trained. See the
[protocol](docs/experiments/internal-rule-discovery.md) before running final evaluation.
The [first frozen result](reports/internal-rule-discovery-v1/README.md) reaches 76.53%
reserved-input accuracy on 100 distinct finite rules, versus 66.60% random queries
and 28.20% after model reset. This is evidence of bounded model learning, with
supplied primitives and search/inference, rather than learned general reasoning.

The earlier graph demo remains a supporting baseline:

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

## Delayed-question memory experiment

The newer controlled experiment withholds the specific future question, shuffles all
observations, and gives memory a limited consolidation budget. Episodic evidence
expires before the question. A neural policy chooses WAIT/CONSOLIDATE and retention
priorities, bootstrapped from a supplied graph heuristic and optionally refined with RL.
Possible question vocabulary/distribution and the horizon are public; this is not
open-ended memory or learned logic. Read the [protocol](docs/experiments/delayed-goal-memory.md).

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner delayed-memory --config configs/delayed-memory-smoke.json --output runs/delayed-smoke --phase validation
```

Use `configs/delayed-memory.json` for the full three-seed experiment. Validation is
the default phase. `--phase final` opens the reserved test range: run it only after
freezing the experiment, and do not tune on those results. Output includes learned,
imitation-only, random, untrained, FIFO, and graph baselines, per-world metrics, a
storage upper bound, and an explicit success/failure gate. A failed gate is a valid
research result and does not mark the milestone complete.

The [first delayed-question report](reports/delayed-memory-v1/README.md) passed its
scoped gate: 22.61% mean learned accuracy versus 4.44% FIFO, with a 33.33% offline
storage-only upper bound. This is a harder task than the original 100% demonstration;
the accuracy figures are not directly comparable across those task settings.

The [frozen robustness report](reports/delayed-robustness-v1/README.md) tests eight
timing/resource scenarios without retraining. It exposes a complete failure with
observation gaps and a loss to scheduled graph with more writes. To reproduce using
the original final checkpoints:

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner delayed-robustness --checkpoints runs/delayed-final-v1 --output runs/delayed-robustness-v1
```

Use a fresh output directory. All published final ranges are already opened; reserve
new worlds before evaluating a revised policy.

## Bounded autonomous investigation

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/session.json --checkpoint runs/experiment/controller.pt --ticks 2
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/session.json --checkpoint runs/experiment/controller.pt --ticks 8
```

The second invocation resumes the first. The loop persists question history and
budgets, uses application timestamps, and stops after answering or exhausting its
resources. Audit traces are separate from accessible memory. Use one writer per
session. Atomic session-file replacement protects against partially written JSON;
the session and evidence database are not one crash-atomic transaction. Schema-v2
sessions bind controller weights, memory capacities/contents, teacher implementation
type, task, and budgets. Missing or mismatched memory and interrupted ticks are
refused before further teacher calls. Legacy sessions must stay with their original
source; use fresh files to start a new run. Existing memory is never trimmed on open
to accommodate a changed capacity. See the [critical audit](docs/CRITIC_AUDIT.md)
for repaired defects, recovery limits, and unresolved research criticism.
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
