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

## Environment and version control

- Local virtual environment `.venv`, Python 3.12.14; see README to recreate elsewhere.
- Installed PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.5.4, Transformers 4.57.6.
- PUBLIC remote: https://github.com/MGG-92/Autonomous-Learner; branch `main` tracks origin.
- `gh` authenticated as `MGG-92`. Scope credentials remain in the OS keyring, not files.
- Repository-scoped Git identity uses the public GitHub no-reply email.
- The private bootstrap commit was sanitized before public publication. Never rewrite
  public history or restore the obsolete bootstrap hash.
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
- 20 local tests pass. Actual trained-controller restart and dual-memory CLI smokes passed.

## Measurements to date (development runs, preliminary)

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

## Immediate next actions

1. Commit the tested tick/language integration, then run the frozen confirmatory
   config over optimizer seeds 7/42/97 on NEW world seeds 500000..500199. Keep the
   configuration unchanged even if a seed underperforms; publish the observed result.
2. Rerun the memory experiment from clean committed source; evaluate integrated
   learned controllers, and preserve sanitized reports under `reports/`.
3. Update this file with results/source hashes, publication check, push, verify CI.
4. Next research increment: remove the helpful fixed proof-before-distractors ordering
   and supplied future-query assumption. Build delayed goals with memory pressure,
   learn consolidation/admission timing, and test against budget-matched heuristics.
   Define and freeze validation/test splits before tuning. See `docs/PLAN.md`.

## Commands for continuation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
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
