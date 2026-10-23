# Recurring critique and repair log

Cadence and merge rules live in [AGENTS.md](../AGENTS.md). Review relevant open
findings on each continuation; record a focused audit before each meaningful PR,
repeat after substantive repairs, and broaden at milestone/final-experiment gates.
This is part of foreground development, with no scheduled job. Entries describe
checks actually performed and never certify that the whole system is flaw-free.

## 2026-10-23 — Initial integrity audit (PR #4)

- Source: parent `054181f`, repairs `4cf3be9`, merged as `ce9c632`.
- Scope: evidence storage, session identity/budgets, interrupted external actions,
  snapshot/configuration validation, paired statistics; adjacent boundary: teacher input.
- Findings/repairs: nine defect categories, with seventeen initial failing
  reproductions. See [detailed audit](CRITIC_AUDIT.md). Added 31 regression cases.
- Evidence: 64 tests passed on Python 3.11/3.12 CI; normal heuristic and actual trained
  dual-memory CLI restarts worked. Question smoke 30/30; short delayed validation
  smoke retained its negative learning gate. Publication preflight passed.
- Open: learned timing generalization, source reliability, automatic crash recovery,
  joint training, and delayed-controller integration. A pending external action is
  refused, not automatically recovered. Opened final ranges remain barred from tuning.
- Next focus: delayed memory in the persistent loop, especially expiration on empty
  ticks, provenance, compatibility, and observer-history isolation.

## 2026-10-23 — Paused delayed-loop integration

- Source: branch `codex/delayed-loop-and-recurring-audit`, parent `ce9c632`.
- Checks: 76 local tests passed; a real frozen-controller demo resumed after two
  ticks and answered after five, using two consolidations. Original seeded delayed
  validation metrics/curves/selections remained unchanged, including its negative gate.
- Counterexample: the parent snapshot loader accepted different facts at the same
  observed tick, which would collide under observation-based proof IDs. Reproduced
  against the parent source; repaired loader rejects it. Tests also cover expiration
  before selection, source preservation, clock mismatch, and interrupted memory commits.
- Status: preserved locally, not published or merged. The user clarified the central
  objective before publication; further integration/viewer work is deferred while
  internal model learning becomes the priority. No new reasoning capability is claimed.
- Next focus: rule induction, model-dependent predictions, active experiments, and
  separating supplied search machinery from experience-driven rule learning.

## 2026-10-23 — Cognitive-core reorientation and C1 source freeze

- Source: `codex/internal-rule-discovery`, parent `ce9c632`; paused integration stays
  on its separate local branch. The recurring audit workflow is retained here.
- Scope: public hypothesis generation, oracle/prediction boundary, reset intervention,
  semantic task splits, and adjacent evaluation/statistics/communication claims.
- Counterexample: four information-gain queries of a max-operation world leave six
  compatible functions. An initial test incorrectly demanded certainty. The corrected
  test requires retained ambiguity, no uniquely identified representative, and correct
  highest-probability predictions on that fixture. No model/hyperparameter was tuned.
- Fairness repairs in design: quotient equivalent expressions before splitting;
  use independent layouts and the same reserved input pairs for every policy;
  commit predictions before evaluator targets; erase raw responses on model reset.
  Mark reset/no-memory equivalence rather than counting them as independent evidence.
  A split audit found four named reference functions hashed into the reserved final
  bucket. Before opening final tasks, all nine named software/example functions
  were assigned development-only. The resulting split is 846/165/199, with no
  equivalent functions shared. Validation results were unchanged; no final result
  informed this correction. The regression checks reference-function exclusion.
- Checks: 77 local tests passed; isolated core import excludes Torch/Transformers/LM
  modules; oracle-call tests enforce legal queries and budgets. Legacy demo passed.
  Full validation: 100 distinct rules, query seeds 7/42/97; active accuracy 72.6%,
  random 66.8%, fixed 54.0%, reset/no-memory 25.6%. These are development evidence,
  not opened final results. Code/config/protocol freeze precedes final evaluation.
- Open: supplied primitives, grammar, inference and heuristic; no learned search,
  concept invention, uncertainty calibration, noise revision, or family extrapolation.
  Query costs are matched but compute costs are not. The initial result can support
  bounded model learning only. Next audit: C2 proposal/search learning under matched
  observation/compute costs and a fresh final protocol.

## 2026-10-23 — Frozen C1 result review

- Source: `aedf284f0bf84a4accf3ccaab827f9a6f8af728d`, clean tree; protocol/config
  committed before opening final functions. No algorithm changes after evaluation.
- Evidence: 100 distinct rules, 500 related held-out inputs, query seeds 7/42/97.
  Accuracy: active 76.53%, random 66.60%, fixed 54.40%, reset/no-memory 28.20%.
  Full-function identification: 25% of active rule/seed runs. The complete report
  includes proper scores, exploratory rule-bootstrap intervals, and an actual trace.
- Review: all 1500 rows checked against aggregates, query limits, unique pairs,
  semantic partitions, validation disjointness, and clean source metadata. No final
  rerun or tuning. No-memory/reset equivalence and unmatched compute remain explicit.
- Interpretation: bounded internal model learning is supported; discovery of a
  reasoning algorithm, new primitives, calibrated uncertainty, and general cognition
  are not. Selected final functions are now opened; use a fresh protocol for C2.
- Next audit: development-only learned search/query proposals, compute accounting,
  function/family splits, and intervention tests that challenge model dependence.

## 2026-10-23 — World-learning roadmap review

- Scope/source: documentation after C1 merge `762ef7f`; reviewed the plan and
  architecture against the adjacent frozen C1 evidence and public README claims.
- Counterexample to a broad interpretation: a learner could perfectly predict every
  function in the supplied grammar while lacking any state-transition model,
  action-sequence planning, or transfer across world mechanisms. C1 does not test
  those abilities. No new software defect or capability result was established.
- Repair: made world learning the explicit destination, kept bounded C2 search
  next, and added a small structured action-world bridge before prolonged arithmetic
  tuning. Added whole-world/mechanism splits, goal tests, transfer/removal controls,
  and a recurring relevance check. Supplied planning and representation priors must
  be reported separately from learned dynamics and abstractions.
- Checks: reviewed documentation diff; `git diff --check` passed; local pytest
  passed all 77 tests; fresh legacy demo passed restart and memory ablation.
  No final experiment was rerun and no frozen result/protocol was modified.
- Open/next audit: C2 still needs a development-only policy design, fair compute
  accounting, and fresh final selection. Before implementing the bridge, specify
  minimal hidden mechanisms and whether proposed observations leak effect labels;
  distinguish causal intervention tests from observational prediction, and genuine
  mechanism transfer from renaming invariance. All bridge/abstraction items remain open.
