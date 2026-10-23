# Project state — 2026-10-23

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
experiment source `d675f4b`. The full final run is complete, the predefined scoped
gate passed, and 31 local tests passed. The current documentation increment is
`docs/memory-activity-viewer`, adding the user-requested M2.5 objective. Inspect its
PR/merge state when resuming and finish any interrupted publication first. The user
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
- 31 local tests pass. Actual trained-controller restart and dual-memory CLI smokes passed.
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
source reliability, changing-horizon/budget evidence, and joint question-policy training.

## Immediate next actions when the user says continue

1. Read this file, `AGENTS.md`, `docs/PLAN.md`, and `reports/README.md`; check Git status
   and the current public remote/CI. Run the current test suite using the local venv.
2. Finish any interrupted current-branch PR/publication first. Then start a NEW
   feature branch. Next research increment: test the frozen delayed-memory policy on
   changing horizons, observation order/delays, and memory/write budgets. Reserve new
   worlds in the seed ledger before evaluation; keep old final worlds out of tuning.
   Use failures to design broader training distributions on separate training and
   validation ranges. Do not claim M2 complete from the single fixed-budget experiment.
   Subsequent work: delayed-memory integration into the persistent loop, followed by
   the M2.5 memory/activity viewer at its documented dependency gate; learned retrieval,
   source reliability/revision, and joint training remain M2 research objectives.
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
