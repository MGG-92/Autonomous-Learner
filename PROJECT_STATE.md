# Project state — 2026-10-23

## Current priority after vision clarification

The user clarified that the goal is an independently learning cognitive core:
experience -> internal models/hypotheses -> experiments/predictions/revision ->
structured communication, with a pretrained LLM only as an optional language
interface. Existing graph inference is programmed, not learned reasoning. Follow
C1–C4 in `docs/PLAN.md` before further memory optimization, token control, or viewer
polish. Do not drift toward an LLM-centered agent or claim prior-free learning.

Broader objective: world learning with objects, properties, actions, interactions,
consequences, and goals. Mathematics is a diagnostic, not the destination. C2's first
bounded comparison is complete; build the small structured action-world bridge in
`docs/PLAN.md` before prolonged arithmetic optimization. Test mechanism
transfer and acquired abstractions separately from renamed-object generalization.
The action-world core now updates transition beliefs within supplied templates;
its first development prediction comparison is complete. Goal planning, learned
investigation, abstraction acquisition and transfer remain unmeasured/unimplemented.

Self-modeling is now an explicit long-term objective: learn reliability, capability
limits and effects of cognitive operations from ordinary experience, then test
behavioral benefit. It is not implemented and implies no consciousness claim.
Keep goal-directed world learning next, then a small C5a reliability pilot alongside
later abstraction/strategy work. Language rendering is not a prerequisite.

## Latest increment — Interactive Agent Playground roadmap

Added `docs/PLAYGROUND.md` and P1 in `docs/PLAN.md`. This is planning only: no
`playground` command, event recorder or interactive action-world session exists yet.
After the first scored goal baseline, take one bounded CLI increment using the real
simulator, transition beliefs, investigation controls and planner. Keep C5a/C3 research
moving; graphics, natural language and complete memory integration are not prerequisites.
The earlier M2.5 graphical viewer is retained as a later extension with shared events.

P1 targets structured goals/questions, step/run, evidence/hypothesis inspection,
model-relative uncertainty and committed plan execution. Actual-operation telemetry
must distinguish belief updating, supplied algorithms, trained parameters and any
later LLM interface. Inspection/replay cannot rerun choices, use oracle calls or
modify work counters; requested predictions are explicit, separately counted work.
Observer history is never learner memory. Existing memory/abstraction/self-model
views are added only as those components become integrated and demonstrated.

Manual sessions stay local under ignored `runs/playground/`, using development
mechanisms only; no reserved targets/loaders, automatic training ingestion or benchmark
promotion. Rule editing is deferred: later edits need semantic partition checks and
logged operator interventions, without secretly resetting/revealing the agent's model.
Current stationary filtering can fail after rule changes; adaptive revision is unproven.

Validation commands for this documentation increment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/playground-roadmap-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner action-model-demo --output runs/playground-roadmap-action-demo.json
.\.venv\Scripts\python.exe scripts/publication_check.py
git diff --check
git diff --cached --check
```

All 127 tests passed in 13.63 seconds; both demos, diff whitespace checks and
publication preflight passed. The action-model demo remains scripted, not an
interactive agent or benchmark result. Playground acceptance checks are future
requirements, not tests already implemented. Latest critique: `docs/AUDIT_LOG.md`, playground planning entry.
Immediate next task is still the development-only scored goal evaluator with independent
goals, prior/random-plan controls, public commitment and source/config freeze; then
P1 as a small observability increment alongside the C5a reliability pilot.
Independent local `docs/PLAN.md` formatting remains preserved outside this commit.

## Previous increment — self-modeling roadmap and supplied planning primitive

Added `docs/SELF_MODELING.md`: critical mapping of existing components, information
boundaries, candidate experiments and research sources. New C5 stages: empirical
reliability forecasting; agent-specific capability/process interventions; adaptive
metacognitive control; and links to learned strategies/abstractions. Compare no-self,
simple calibration, task-only, matched-resource/history-based and monitor-only
controls, including shuffled/stale self histories. No canned introspective statements,
LLM self-awareness simulation or subjective-consciousness result is planned/claimed.

Existing predictive uncertainty, memories, resets and strategy/cost records are
potential inputs, not learned self-understanding. Any future agent-visible internal
event interface must be distinct from inaccessible research/audit logs, omit hidden
condition labels/gold targets and not retain forgotten task contents off budget.
Monitor training feedback must come from explicitly available normal experience;
the opened evaluator panels cannot be silently reused as learner experience.
Forecasting improvements alone do not establish control benefits or an agent-specific
self-model. Task difficulty, extra resources and strategy-selection bias need controls.

Continued the existing development order with `action_planning.py`: supplied exhaustive
open-loop search of sequences up to three actions, using only acquired dynamics and
public start/goal. Prefer predicted success, then shorter plans, then public RNG ties.
It reports simulated work, consumes no oracle feedback, and leaves learned evidence
unchanged. Six new software cases cover an experience-grounded chain plan, committed
execution, ties/shorter/empty plans, work counts, model failure and bad budgets.
Predicted success is task-model-relative probability, not learned self-reliability.
The primitive is not wired into a new agent policy or goal experiment yet; no new
policy performance, goal benchmark or metacognition result is claimed.

Current suite: 127 tests passed. Baseline suite was 121 before this increment.
Commands used for software checks (choose fresh demo outputs):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_action_planning.py -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/self-model-roadmap-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner action-model-demo --output runs/self-model-roadmap-action-demo.json
```

Latest audit: self-modeling roadmap/planner primitive in `docs/AUDIT_LOG.md`.
No old config, controller, frozen protocol or report changed. Next: integrate a
development-only goal evaluator with independent goal/start sampling, prior and
random-plan controls, public commitment before feedback and separate reachable-goal
analysis outside the learner. Freeze source/config before scoring; only after that
baseline specify the C5a internal-feedback schema and reliability-learning pilot.
Independent local formatting changes in `docs/PLAN.md` (heading/checkbox formatting)
were observed during this work and preserved outside the commit. Do not discard or
silently stage them on continuation; the PR contains only the intended roadmap additions.

## Preserved scored action-world prediction comparison

Frozen clean source/config/protocol: `b94d48acb3880e88e244f093cf196ef073cfef3f`.
Implemented `action_investigation.py` controls (supplied one-step information gain,
random and fixed cyclic actions), and `action_prediction.py` for development-only
joint prediction scoring. 80 canonical development worlds, 12 real actions per run,
24 one-step/16 two-step probes, policy RNG seeds 7/42/97. All policies/seeds commit
all predictions before any panel target access. Shared unseen cases exclude the
union of every compared trajectory across policies AND seeds. Empty panels stay
unavailable; macro scores and bootstrap denominators are explicit. No final phase.

Full-panel seed-mean accuracy: information gain 75.31% one-step / 71.80% two-step;
random 73.63% / 69.97%; fixed 73.33% / 69.84%; reset/prior 61.98% / 56.41%.
Information-gain Brier 0.3407/0.3837 versus random 0.3672/0.4131 and reset
0.5682/0.6074. Shared-unseen Brier 0.4326/0.4767; reductions versus random
+0.0173 [0.0087, 0.0274] / +0.0223 [0.0110, 0.0355], versus reset
+0.1415 [0.1093, 0.1723] / +0.1365 [0.1018, 0.1717]. Intervals bootstrap
worlds after averaging seeds, exploratory/unadjusted. Two-step unseen versus fixed
+0.0105 [-0.0055, 0.0289] crosses zero; no universal investigation advantage.

Shared-unseen coverage: mean 17.325/24 and 12.075/16; no empty panels this run.
Of 966 unseen two-action probes, 47 consist of single transitions seen somewhere
in the union. Stasis is a strong baseline: the prior argmax always predicts no
change, correct on 1190/1920 one-step and 722/1280 two-step targets. This post-hoc
interpretation check uses saved raw results, without tuning or new world execution.
Information gain averages 9.0625 no-ops/12 actions. Interaction counts are matched;
compute is not: exploration effect evaluations 1043.1 versus random/fixed 293.6/291.3.
Inference, templates and investigation remain supplied; this is experience-conditioned
model learning, not trained reasoning, primitive discovery, planning or transfer.

Report: `reports/action-prediction-v1/README.md` and `development.json`. All 3840
score rows, 960 cost records, raw predictions/targets, shared-unseen sets, aggregates,
paired differences and bootstrap intervals verified offline without rerunning worlds.
Effect-evaluation counts independently reconstructed from public traces. Only the
reviewed compact synthetic report is published; raw files stay in ignored `runs/`.
That increment's audit: prediction-stage in `docs/AUDIT_LOG.md`. 121 tests passed, plus
fresh legacy/model demos and prediction smoke. Exact commands (fresh paths):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/action-prediction-start-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner action-model-demo --output runs/action-prediction-model-smoke.json
.\.venv\Scripts\python.exe -m autonomous_learner action-prediction --config configs/action-prediction-smoke.json --output runs/action-prediction-software-smoke-v1
.\.venv\Scripts\python.exe -m autonomous_learner action-prediction --config configs/action-prediction.json --output runs/action-prediction-v1
```

Next: implement supplied open-loop goal planning from acquired dynamics, independent
goal/start sampling, prior/no-memory and random-plan controls, with a new scope/config
freeze before measuring. Keep source-stable previous prediction results intact;
do not silently convert development outcomes into final or family-transfer evidence.

## Preserved transition model and development sampler

Implemented `action_model.py`: independent uniform priors over 45 public effects
per action, conditioned only on experienced full-state transitions. Predictions
integrate fixed hypothetical effects across an entire sequence (up to three actions),
not independent resampling at each step. True effect instances and mechanism IDs
stay outside the core. Reset erases posterior restrictions, raw observations and
work counters. Contradictions yield explicit global model-class failure.
This is actual evidence-conditioned belief learning with supplied inference and
primitives; no parameter optimization, new vocabulary or learned search is claimed.

`action_sampling.py` enumerates 2744 unique canonical mechanisms and samples only
development identities uniformly without replacement. Partitions: 1934 development,
410 validation, 400 future final, including the two named-fixture exclusions.
Episode slots are independently permuted before public naming; execution rejects
non-development or forged identities. Enumeration uses supplied hypotheses to define
splits; it is not final-world execution or performance evaluation. Cached public
effect tables preserve the original canonical hash (checked against uncached code).
The learner's independent labeled-effect prior is NOT uniform over these canonical
worlds. Model-relative probabilities must not be called population calibration.

Added `action-model-demo`: five scripted development-fixture actions, public trace,
before/after/reset predictions, seeds and source metadata, preserved in ignored
`runs/`. The demo records 27/6/3 remaining effect candidates by action and restores
the exact prior on reset. It does not score a probe against the oracle, train an
investigation policy, achieve goals or measure transfer. Full protocol/config stays
development-only; no full experiment runner or final phase is implemented.

That increment's audit: transition-model/sampler in `docs/AUDIT_LOG.md`. Checks: 111 tests
passed, including nine new cases. The decisive rollout fixture observes 100 -> 000:
fixed set-zero/toggle hypotheses imply 000 after two actions from 000, whereas the
incorrect independent-step approximation assigns 25% to 100. True hypotheses are
retained over 12 real actions in eight development fixtures; illegal transitions
are rejected before mutation, and misspecification/reset are explicit. Pure core
imports no Torch/Transformers/language/sampler modules. Legacy demo also passed.
The adjacent record-validation gap was reproduced from committed source: an illegal
action and mixed catalogs were previously accepted; the new validator rejects both.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_action_model.py -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/action-model-start-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner action-model-demo --output runs/action-model-software-demo-v1.json
```

Next: add controlled investigation and prediction scoring on development worlds,
then supplied open-loop goal planning. Keep all baselines, actual seen/unseen panel
coverage and real/simulated cost counters; freeze source/config before recording
performance. Do not promote scripted examples or software fixtures into benchmark
claims. C1/C2 results and paused local work are preserved.

## Preserved action-world development contract

Added `docs/experiments/structured-action-world.md` and proposed settings in
`configs/action-world-development.json` (protocol-only, no runner). Implemented
`action_world.py`: three objects with binary properties, three actions with hidden
guarded set/toggle effects, immutable public observations/transitions, hard paid
action budgets including no-ops, and no agent-accessible reset. Public hypothesis
templates and evaluator-only true mechanisms are distinct responsibilities.

Semantic identity canonicalizes whole transition behavior under object/action
renaming. The future sampler must split unique mechanism identities, reserve named
fixtures for development, and independently shuffle slots before naming episodes;
none of that sampler or any final evaluation is implemented yet. The contract
specifies consistent joint rollouts, committed prediction panels with explicit
seen/unseen coverage, and later supplied planning from acquired dynamics. Goal
selection must be independent of oracle reachability. No learner, learned strategy,
planning success, general reasoning or transfer is demonstrated by this increment.

That increment's audit: action-world contract in `docs/AUDIT_LOG.md`. Checks: 102 tests passed,
including 14 new cases for order-sensitive interactions, no-op budgets, rejected
calls, immutable label-free public records, distinct effect hypotheses and all 36
object/action renamings. Legacy demo and C1 smoke passed. Commands (fresh paths):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_action_world.py -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/action-world-start-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery-smoke.json --output runs/action-world-rule-smoke
```

At that increment the next task was the development-only mechanism sampler and experience-updated
transition model; test true-hypothesis retention, erasure, and fixed-mechanism joint
rollouts before adding a planner or claiming measured world learning. Preserve all
existing frozen reports; do not tune arithmetic or open final worlds to bypass the
mixed C2 result. Proposed 80-world/12-action/three-step settings are not measurements.

## Preserved learned investigation development comparison

First C2 query policy implemented and evaluated at clean source/config/protocol
freeze `bcd7f5106425b18db217c049e0cb8a09435b20f0`. Shared 7 -> 16 -> 1 scorer uses
sorted predictive masses, model count, and budget. REINFORCE trains from Brier
prediction improvement on development functions, without query demonstrations,
entropy labels, or language. Exact inference, grammar, features, reward, and optimizer
are supplied. Last episode is selected by a fixed 800-episode schedule, not validation.

Optimizer seeds 7/42/97; 80 shared validation functions, five reserved inputs, four
queries. Mean trained accuracy 75.0%, random 70.25%, initial networks 75.83%,
information gain 76.5%. Trained Brier 0.2895/0.2787/0.3196 versus initialization
0.2866/0.3404/0.2656; benefit over initialization is not consistent. Information
gain Brier 0.2658 beats every trained seed. Updated model access beats prior-feature
selection in all seeds; prediction reset/no-memory gives 26.25% accuracy. Compute
is measured, not matched: trained/heuristic selection both scan about 30000 candidate
outputs/world; no speedup demonstrated. This is mixed development evidence, not a
final result or proof of learned reasoning. No final functions were evaluated.

Report: `reports/learned-investigation-v1/README.md` and `validation.json`. Verified
all 1680 local raw rows and paired differences against aggregates without rerunning.
Checkpoints/raw rows remain in ignored `runs/`. At that increment, 88 tests passed; fresh
legacy demo, C1 rule smoke, and new query-policy smoke passed. Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/c2-start-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery-smoke.json --output runs/c2-regression-rules-v1
.\.venv\Scripts\python.exe -m autonomous_learner learned-investigation --config configs/learned-investigation-smoke.json --output runs/c2-software-smoke-v1
.\.venv\Scripts\python.exe -m autonomous_learner learned-investigation --config configs/learned-investigation.json --output runs/learned-investigation-v1
```

Its recurring audit is C2 first development comparison in `docs/AUDIT_LOG.md`.
The structured action-world development contract is now specified above. Keep C2
compute matching, mechanism/family transfer, and reliable training benefit open;
do not tune indefinitely for arithmetic gains or pick a favorable seed. The current
sorted-marginal features cannot distinguish equal marginals with different joint
action consequences. Critique that representation and prevent hidden-effect leakage
before implementing the bridge. C1 final results/protocol and saved branches remain intact.

## Preserved C1 and supporting work

The user authorized applying this direction. Paused delayed-loop integration and
the recurring audit workflow were preserved locally in commit `cda82bf` on branch
`codex/delayed-loop-and-recurring-audit`, not pushed or merged. Its 76 tests and real
checkpoint demo passed, but it is supporting engineering, not evidence of cognition.
Do not delete that branch or cherry-pick its whole implementation merely to resume.
The recurring workflow/log and C1 were merged through PR #5 at `762ef7f`.

C1 implementation: generated depth-two expressions with modular arithmetic/min/max
primitives, quotiented into 1210 distinct finite functions; observation-conditioned
version-space predictions; supplied information-gain investigation; random and
fixed/untrained controls; no-memory and model-reset interventions; semantic rule
splits; independent reserved input layouts. Core imports no Torch/Transformers/LM
module. This learns rules within a supplied representation by belief updating. It
does not train parameters, invent primitives, or learn a reasoning/search strategy.
Protocol: `docs/experiments/internal-rule-discovery.md`. The selected final functions
are now OPENED after clean source/config freeze `aedf284f0bf84a4accf3ccaab827f9a6f8af728d`.
Keep these and the old graph final ranges barred from tuning.

At the C1 source freeze, 77 local tests passed, including core import without Torch/Transformers/language
modules, budget/observation boundaries, reset causality, semantic splits, and explicit
model-class failure. Legacy demo also passed. Full validation on 100 distinct rules:
active 72.6%, random 66.8%, fixed/untrained 54.0%, reset/no-memory 25.6% reserved-input
accuracy. An initial overconfidence assumption was falsified: four observations of
max leave six hypotheses in the regression fixture. Do not claim unique discovery
from correct predictions or probabilities as calibrated real-world confidence.

Commands used before the source freeze (choose fresh paths to repeat):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery-smoke.json --output runs/rule-discovery-smoke-v1
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery.json --output runs/rule-discovery-validation-v1
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery.json --output runs/rule-discovery-validation-v2
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/rules-reorientation-demo.sqlite
```

Frozen final command:
`.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery.json --output runs/rule-discovery-final-v1 --phase final`.
Clean source, 100 distinct final functions, five reserved inputs per function,
query seeds 7/42/97 (exploration repeats, not optimizer seeds). Active accuracy
76.53%, random 66.60%, fixed/untrained 54.40%, reset/no-memory 28.20%. Brier scores
0.2460/0.4007/0.5360/0.7804 respectively. Active full-function identification is
25% of rule/query-seed runs. Paired Brier reduction versus random +0.1547,
rule-bootstrap95 [0.1086, 0.2007]; versus reset +0.5344 [0.4960, 0.5706].
Exploratory/unadjusted intervals, seeds averaged within rules. Compute is not matched;
no-memory and reset are equivalent controls. All 1500 synthetic result rows and
aggregates were reviewed without rerunning final worlds. Report:
`reports/internal-rule-discovery-v1/README.md`. Total evaluation after hypothesis
construction: 1.27 seconds locally; no portable performance claim.

C1 is a measured bounded rule-learning baseline, not learned inference or concept
invention. C2's first development comparison is recorded above; any eventual final
evaluation requires a NEW reserved protocol. Preserve the recurring
audit and the local paused integration branch. No LLM/token/viewer development
should displace the internal-model research objective.

World-learning roadmap clarification: documentation only; no learner, protocol,
frozen report, or final-world selection changed. Local checks passed:
`git diff --check`, `.\.venv\Scripts\python.exe -m pytest -q` (77 tests), and
`.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/world-roadmap-demo.sqlite`
(restart retained evidence; memory ablation returned UNKNOWN). Publication preflight
also passed. No new capability measurement is claimed.

## User intent and authorization

Implement the architecture in `docs/ARCHITECTURE.md` incrementally and autonomously.
Train, evaluate, debug, keep appropriate Git history, and maintain a PUBLIC GitHub repository.
The user clarified public visibility and emphasized careful publication. Use the
publication check, GitHub no-reply author metadata, and explicit source-only staging.
The README must prominently disclose work in progress and that most development
is done by agentic AI. Present the project as an experiment, nothing less and nothing more.
Make progress across sessions; the user will say “continue” after usage refills.
The original attached draft is inspiration; the accepted architecture prioritizes
controlled experiments before language/token control. Never claim all milestones
complete merely because a prototype runs.

New user objective: a simple, visually appealing, restrained 3D representation of
memory and agent activity. Scheduled as M2.5 in `docs/PLAN.md`, after robustness
testing and delayed-memory integration into the persistent loop, before expanded
M3 investigation. Start with a local read-only graph, fact inspection, activity
timeline, and saved-session replay; then add live observation. This is planned,
not implemented. Preserve real forgetting and keep observer history out of policy.
The user clarified the teaching objective: show which parts of the agent activate
as it works. Prioritize operation-level recording and slow replay, with a simple
component view alongside the 3D memory graph, plain-language explanations, and actual
controller scores. Distinguish trained decisions from supplied inference. Optional
neural activations must be measured and labeled, never invented or presented as
thoughts/confidence. Playback must not rerun decisions or change agent behavior.

## Environment and version control

The delayed-memory increment in https://github.com/MGG-92/Autonomous-Learner/pull/1
was merged as `3f1f5dd`. Source review/local checks and PR CI passed for frozen
experiment source `d675f4b`. The full final run is complete and the predefined scoped
gate passed. The viewer-planning PR #2 merged as `da38854`; main CI passed.
The robustness increment merged through PR #3 as `054181f`, after both Python CI
jobs passed; evaluation source remains frozen as `ebd2099`. The critic-audit increment
merged through PR #4 as `ce9c632`, with both Python CI jobs passing. Current work is
the cognitive-core reorientation described above. The user
requires feature branches and PRs for meaningful changes; do not develop directly
on main. Preserve source hashes with merge commits and GitHub no-reply metadata;
do not squash/rebase experiment commits referenced by reports. Keep main stable.

- Local virtual environment `.venv`, Python 3.12.14; see README to recreate elsewhere.
- Installed PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.5.4, Transformers 4.57.6.
- PUBLIC remote: https://github.com/MGG-92/Autonomous-Learner; branch `main` tracks origin.
- `gh` authenticated as `MGG-92`. Scope credentials remain in the OS keyring, not files.
- Repository-scoped Git identity uses the public GitHub no-reply email.
- Bootstrap publication correction: rewriting a branch did NOT purge GitHub's old
  commit object. The original remote was made private and renamed to
  `Autonomous-Learner-private-bootstrap`; retain that archive PRIVATE. A new clean
  PUBLIC repository was created at the original project URL, and only sanitized
  reachable history was pushed. GitHub API verified the obsolete object is absent
  from the new public repository. Never mirror-push local refs, restore obsolete
  bootstrap objects, or make the private archive public. No user work was deleted.
  Anonymous HTTP checks confirmed public repository 200, private archive 404, and
  obsolete commit in the clean public repository 422 (absent).
- GitHub Actions passed the initial controlled and memory experiments on Linux.
  Check the current revision's CI after each push.

## Implemented and verified

- Randomized synthetic worlds; oracle answers unavailable to policy feature encoding.
- Bounded SQLite evidence, provenance, timestamps, proof IDs, positive/negative/unknown/
  conflicting capability evidence. Supplied monotonic inheritance inference.
- PyTorch question/action scorer; supervised expert bootstrap and REINFORCE fine-tuning;
  validation-only checkpoint selection; random/untrained/heuristic/no-memory baselines.
- Separate bounded episodic and belief graphs; supervised retention scorer versus
  FIFO/untrained/heuristic. Fixed consolidation timing, hard capacities (4 and 3).
- Restartable autonomous ticks, saved question history, timestamps, compute/teacher
  budgets, idle on completion. Human JSON teacher and separate unverified local-LM teacher.
- Integration of separately trained question and retention controllers in the tick loop.
- Optional frozen SmolLM2 adapter, entity copying, strict faithful template fallback.
- Cached candidate pagination (200 per page), trainable residual token scorer, MORE/END,
  supervised-loss interface, and real token-selection generation driver.
- Researcher memory inspection and Graphviz DOT export; audit logs excluded from policy.
- 77 local tests pass. Actual trained-controller restart and dual-memory CLI smokes passed in the earlier track.
- Delayed-specific-goal benchmark: shuffled facts, expiring episodic evidence, neural
  WAIT/CONSOLIDATE gate and retention scores, persistent bounded snapshots, graph/FIFO/
  random/untrained/imitation controls, predefined gate, and per-world resampling.

## Measurements (preliminary; published reports available)

M1 seed 42: learned and heuristic 100/100 held-out worlds; random/untrained/no-memory
0/100. Both successful policies ask 4 questions. No advantage over the heuristic;
no additional greedy-accuracy gain established for RL after imitation. Full defaults:
160 imitation worlds, 12 epochs, 160 RL episodes. The 40-world smoke requires 24
epochs; the initial 6-epoch short test failed due to too few optimizer updates.

M2 seed 42: learned/heuristic retention 100/100; FIFO/untrained 0/100. Future query
is supplied as a hint, and proof evidence precedes distractors. This is a narrow
retention experiment, not learned retrieval or consolidation timing.

Real LM smoke: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. All weights frozen; two 200-token pages
used one forward pass. Untrained scorer emitted 8 tokens using the prior. The LM
failed exact sentence fidelity, correctly triggering template fallback. No language
quality benefit established. Cached model lives in ignored `runs/models/`.

Frozen replication is complete from clean source `3666831`: optimizer seeds 7/42/97
all answered 200/200 worlds (500000..500199). Mean teacher calls: 3/4/3, versus
heuristic 4. Random correct counts: 0/0/1; untrained and no-memory correct counts:
0/0/0. Selected checkpoints were imitation epochs 10/5/8, so no added RL benefit
was established. These are 200 shared worlds across three optimizer seeds, not 600
independent worlds. Full reports and validation curves are in `reports/`.

Memory reproduction from the same clean source matches prior results. Integrated
learned question/retention controllers answered 200/200 additional worlds starting
at 600000, using 4 questions. This is deployment integration, not joint optimization.

New M2 delayed-question experiment: see `reports/delayed-memory-v1/README.md`.
Frozen source `d675f4b38749c16b7054644b1ca846beed90d946`, clean tree; test worlds
2000000..2000199 are now OPENED and must never be reused for tuning. Across seeds
7/42/97, accuracy is 25.11%/18.78%/23.94% (mean 22.61%); FIFO 4.44%, strongest scheduled
graph heuristic 19.56%, offline storage-only bound 33.33%. Zero capacity violations.
Mean paired utility improvements: FIFO +0.1745 [0.1609, 0.1868], scheduled graph
+0.0234 [0.0067, 0.0391], imitation-only +0.0399 [0.0314, 0.0486]. Intervals resample
worlds after averaging optimizer seeds; secondary comparisons are exploratory and
unadjusted. One seed underperforms the strong graph baseline. Selected checkpoints
are RL iteration 40, imitation epoch 11, RL iteration 240; the imitation ablation
supports an RL contribution in this particular task. Inference remains supplied.

Negative development result retained: the dynamic graph bootstrap exhausted writes
early and selected idle initial checkpoints. Reproduced from the clean frozen source
using `bootstrap_policy: graph`; all those validation gates failed. Scheduled graph
bootstrap was selected using validation, before final tests. The old reserved 900000
test range was moved before use to avoid demo seed 900001. See protocol/seed ledger.
This scoped experiment passes, but M2 still lacks learned retrieval, general revision,
source reliability, broad timing/resource training, and joint question-policy training.

Frozen robustness v1 now supplies changing-budget/horizon evidence; it does NOT
establish broad robustness. Evaluation source `ebd209995ed0cd6565fdd5923f60baae754185e1`,
clean tree, original final-v1 checkpoints, optimizer seeds 7/42/97. New worlds
2100000..2100099 are OPENED and barred from tuning. Eight scenarios share these 100
worlds. Reference mean accuracy 22.11%; two empty ticks between observations cause
all learned policies to make zero writes and answer 0% (random 4.19%). With nine
allowed writes, learned accuracy 24.59% versus scheduled graph 28.33%; paired utility
difference -0.0291 [-0.0424, -0.0162]. Three belief slots give 7.44% learned accuracy;
nine give 36.63%. Zero capacity violations across all scenarios/policies. Reference
utility advantage over scheduled graph has interval [-0.0073, 0.0435], including zero.
All intervals exploratory/unadjusted, over worlds only. No retraining or selection.
Full sanitized metrics, checkpoint hashes, and limitations:
`reports/delayed-robustness-v1/README.md`. Quiet-tail extension changes policy timing
and heuristic scheduling; beliefs do not decay, so this is not long-term retention.

Validation of this increment: `.\.venv\Scripts\python.exe -m pytest -q` (33 passed),
`-m autonomous_learner demo --memory runs/robustness-demo.sqlite` (passed),
`-m autonomous_learner delayed-memory --config configs/delayed-memory-smoke.json
--output runs/delayed-robustness-regression --phase validation` (completed), and
`-m autonomous_learner delayed-robustness --checkpoints runs/delayed-final-v1
--output runs/delayed-robustness-v1` (completed, 25.27 seconds). The default demo path
already exists after the initial smoke; repeat demos must always use fresh paths.

## Critic audit and reliability repairs

The user requested critical review and fixes. `docs/CRITIC_AUDIT.md` records the
reproductions, repairs, evidence, compatibility break, and remaining research limits.
Seventeen initial regression cases failed on the prior source; the expanded suite
now passes 64 tests. Repairs: non-destructive memory capacity checks/metadata,
atomic observation/eviction storage, strict evidence/config/snapshot validation,
complete paired-statistics validation, and schema-v2 persistent sessions that bind
task/budgets, feature schema, question/retention weights, memory capacities/contents,
and teacher implementation type. ASK intent and budget are saved before the
teacher call. Interrupted or inconsistent ticks stop; automatic recovery is NOT
implemented. Legacy session files are preserved/refused, not silently migrated.
Use their original source or start fresh session/database files. Digests detect
mismatch but are not authentication. One writer remains required.

Exact local checks for this increment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner demo --memory runs/critic-demo.sqlite
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/critic-session.json --ticks 2
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/critic-session.json --ticks 8
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/critic-trained.json --checkpoint runs/replication/seed-42/controller.pt --retention-checkpoint runs/memory-confirmatory/retention.pt --ticks 2
.\.venv\Scripts\python.exe -m autonomous_learner run --session runs/critic-trained.json --checkpoint runs/replication/seed-42/controller.pt --retention-checkpoint runs/memory-confirmatory/retention.pt --ticks 8
.\.venv\Scripts\python.exe -m autonomous_learner experiment --config configs/smoke.json --output runs/critic-question-smoke
.\.venv\Scripts\python.exe -m autonomous_learner delayed-memory --config configs/delayed-memory-smoke.json --output runs/critic-delayed-smoke --phase validation
```

Both CLI sessions resumed after two ticks and answered after five total. The seeded
question smoke answered 30/30; random/untrained/no-memory answered 0/30. The short
delayed validation smoke completed with a negative gate, learned accuracy 0/6;
this is preserved as a software smoke, not a new successful learning result. The
strict paired-statistics implementation reproduces the published reference
comparison exactly from CSV without reopening worlds. No policy logic, final
reports, checkpoint selection, or opened evaluation worlds were changed or tuned.
The learned timing failure remains unresolved and requires new training/evidence.

## Immediate next actions when the user says continue

1. Read this file, `AGENTS.md`, `docs/PLAN.md`, and `reports/README.md`; check Git status
   and the current public remote/CI. Run the current test suite using the local venv.
2. The prediction-stage comparison and supplied planning primitive are complete;
   preserve the report and limits. Integrate the development-only goal evaluator,
   independently sampled goals/starts, prior/no-memory and random-plan controls,
   under `docs/experiments/structured-action-world.md`. Freeze the new scope/config
   before measurement, keep a development-only phase, and report unreachable goals
   outside learner inputs. Retain earlier controls; measure real actions and
   simulated planning work. Do not claim a learned planner or mechanism transfer.
   Then specify the small C5a reliability pilot in `docs/SELF_MODELING.md`; no
   research/audit log or held-out target may become introspection input.
   Keep supplied planning separate from acquired dynamics. Preserve C2's mixed
   training evidence and open compute-matching and transfer questions; do not tune
   on opened final functions or either delayed-memory
   final range. Use new development configs and freeze a fresh final protocol only
   when the developmental hypothesis and baselines justify opening final worlds.
   Recurring critique is required before every meaningful PR; append actual
   findings to `docs/AUDIT_LOG.md`. Defer viewer/token work until the core has models
   and computations worth explaining; retain their designs and saved integration work.
3. Preserve existing reports; add new experiment configs/results under new names.
   Continue documenting unsupported claims and negative results. Do not expand the
   language track merely to avoid a failed controlled experiment.
4. Before pushing: publication preflight, staged diff review, tests, coherent commit,
   push only reviewed `main`/feature branches, verify CI, update this handoff.

## Commands for continuation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m autonomous_learner delayed-memory --config configs/delayed-memory-smoke.json --output runs/delayed-next-smoke --phase validation
.\.venv\Scripts\python.exe -m autonomous_learner replicate --config configs/confirmatory.json --seeds 7 42 97 --output runs/replication
.\.venv\Scripts\python.exe -m autonomous_learner memory-experiment --output runs/memory-confirmatory
.\.venv\Scripts\python.exe -m autonomous_learner language-smoke --local-only --output runs/language-repeat.json
.\.venv\Scripts\python.exe scripts/publication_check.py
```

Choose fresh output paths if these already exist; never delete old runs just to rerun.
Use one writer per session. Session JSON writes are atomic, but session state and
SQLite are not a single crash-atomic transaction. Source-reliability learning,
general belief revision, calibrated uncertainty, learned reasoning, useful internal
THINK operations, token-control benefit, and GPU cost measurements remain unfinished.
