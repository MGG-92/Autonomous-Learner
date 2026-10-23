# Project state — 2026-10-23

## User intent and authorization

Implement the architecture in `docs/ARCHITECTURE.md` incrementally and autonomously.
Train, evaluate, debug, keep appropriate Git history, and maintain a PUBLIC GitHub repository.
The user clarified public visibility and emphasized careful publication. Use the
publication check, GitHub no-reply author metadata, and explicit source-only staging.
Make progress across sessions; the user will say “continue” after usage refills.
The original attached draft is inspiration; the accepted architecture prioritizes
controlled experiments before language/token control. Never claim all milestones
complete merely because a prototype runs.

## Current environment

- Workspace: this checkout (initially empty).
- Local Git initialized on `main`; author identity already configured by user.
- Base Python: Codex desktop bundled runtime; see README's optional bootstrap command.
- Local virtual environment: `.venv`; Python 3.12.14.
- GitHub account: `MGG-92`; user authenticated `gh` after setup began.
- Remote: https://github.com/MGG-92/Autonomous-Learner (initially private; public requested).
- Local repository author/committer now uses GitHub no-reply identity. The sole
  private bootstrap commit was amended to remove personal email and local paths
  before public publication. Do not restore its obsolete hash.
- Installed PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.5.4; editable project installed.

## Implemented and verified

Randomized synthetic world and oracle; legal query catalog without answers;
SQLite bounded evidence; provenance and multi-step proof IDs; unknown/negative/conflict
statuses; feature-based neural action scorer; structured agent loop with step budget;
expert imitation and REINFORCE training; validation checkpoint selection; held-out
baseline evaluation; CLI acquisition/restart/transfer demo; initial tests and CI workflow.

Validation: 9 tests passed. Default diagnostic experiment (seed 42) reached 100/100
test-world accuracy, matching the heuristic; initial/random/no-memory were 0/100.
Learned and heuristic each used 4 questions per task. This easy task has scripted
inference; do not interpret the result as general reasoning or superiority to heuristics.
Demo `runs/first-demo.sqlite` verified negative acquisition, restart, related new query,
and memory ablation. Initial short test training failed; 24 epochs on 40 worlds fixes
insufficient optimizer updates. Full default configuration remains 160 worlds/12 epochs.
Diagnostic runs predate the first source commit; rerun from committed source for reports.

M2 increment: separate bounded episodic/belief SQLite stores; learned supervised
retention priorities; fixed consolidation timing. On 100 held-out delayed-reuse
worlds, learned and heuristic retained 100%, FIFO and untrained retained 0%.
Future query is given as a hint and proof arrives before distractors. Not learned
retrieval, not learned consolidation timing, not yet integrated with asking.
11 tests pass, including learned retention and dual-store restart.

## Next actions

1. Commit M1 source and run default seeded experiment from that revision;
   preserve report/curve under `reports/` and record reproducible environment.
2. Add optional language/token interfaces and begin M2 experiments as practical.
5. Audit source and commit history, replace bootstrap personal author metadata with
   GitHub no-reply identity, push the reviewed source, switch to PUBLIC, verify CI.
6. Continue M2 after M1 evidence and project durability are established.

## Known scientific limits

Inference is scripted, memory admission/eviction is fixed, graph family is simple,
single training seed, no calibrated probabilities, no language model loaded. Future
milestones cover those areas. The purpose of M1 is inspectable knowledge acquisition
and an optimized information-seeking policy, not general intelligence.
