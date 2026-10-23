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

## 2026-10-23 — C2 first development comparison

- Scope/source: query policy, trainer/evaluator, CLI/CI and adjacent C1 model/scoring
  boundary. Frozen code/config/protocol `bcd7f5106425b18db217c049e0cb8a09435b20f0`,
  clean tree before the three-seed run. C1 implementation/protocol/reports unchanged.
- Critical counterexample: parameter updates and beating random do not establish a
  training benefit. Exact initial networks reach 75.83% mean accuracy versus trained
  75.0%; trained Brier worsens in seeds 7/97, improves in 42. All trained seeds lose
  to information gain on Brier. Preserve this mixed result without tuning or selecting
  a favorable seed. Existing supplied features can make an untrained scorer useful.
- Representation limit: identical sorted predictive masses produce identical query
  features despite potentially different joint consequences/future query value.
  Removing coordinates/output names supplies invariance but restricts strategies;
  this is not acquired abstraction or evidence of transferable causal reasoning.
  An executable public-table counterexample confirms identical query features for
  an action redundant with a third query and another independent of it; observing
  the third query makes their distinct relationships visible to the model.
- Integrity checks: training uses only development rule identities and training probe
  rewards; policy input has no truth/program IDs/entropy labels. Validation uses
  disjoint functions/layout seeds and last-episode weights, no selection. Prediction
  targets are read after predictions; legal queries exclude probes. Prior-feature
  ablation removes posterior access specifically, retaining history/budget/common
  stopping. Reset clears posterior and raw observations while preserving queries.
- Actual checks: 88 tests passed; deterministic optimization changes weights and
  preserves the exact initialization; output relabeling leaves features/choices
  invariant; reset gives prior predictions; one-step scan counts match exact costs.
  Fresh legacy demo, C1 smoke and new policy smoke passed. All 1680 development rows,
  aggregate metrics and paired differences were reviewed offline against source
  metadata without rerunning validation. No new final tasks were opened.
- Cost/claims: query ceilings matched, compute measured only. Both trained and
  heuristic selection scan approximately 30000 hypothetical outputs/world; training
  adds approximately 30.5 million reads/seed. No speedup or amortization claim.
  No new correctness defect was reproduced in the adjacent C1 path; its existing
  deterministic finite-class and uncertainty limitations remain explicit.
- Next: minimal structured action-world development protocol; test hidden-effect
  leakage, joint transition representations, goal/planner separation, and mechanism
  transfer. C2 matched-compute/family tests and reliable optimization benefit stay
  open. Do not count documentation of these gaps as completed research milestones.

## 2026-10-23 — Structured action-world development contract

- Scope/source: new evaluator-owned simulator, public records and development
  protocol; adjacent C2 representation limits and C1/C2 split/score boundaries.
  Parent source `2c71b8f`; no old learner, protocol or measured report changed.
- Counterexamples: acting on a guarded object before its prerequisite can consume
  budget without changing state; reversing the order changes the result. A software
  fixture reproduces both. Therefore observations must include pre/post state and
  valid no-ops must count. Tests also reject illegal/exhausted actions before mutation.
- Leakage/splits: two different hidden mechanisms yield identical initial public
  records under the same names/state. No target, guard, operation, mechanism hash,
  reachability or gold path is exposed. Tests cover immutable records and independent
  naming. All 36 object/action renamings share the same behavior identity; a truly
  different transition mechanism does not. Future splits must use this identity,
  not seeds or ID strings, and reserve named software fixtures for development.
- Adjacent representation critique: a canonical mechanism representative can itself
  give public positions stable roles even with random ID strings. The protocol now
  requires independent object/action-slot permutations after identity selection.
  The sampler is not implemented, so this is a preventive design requirement, not
  a repaired observed training leak. Uniform sampling must quotient multiplicities.
- Future evaluator counterexamples: independently resampling effects during a
  rollout breaks fixed-mechanism dynamics; choosing goals from true paths leaks
  reachability; calling all fixed-panel tuples unseen can mislabel familiar transitions.
  The contract requires consistent joint rollouts, independent goal sampling and
  explicit common-unseen coverage (empty means unavailable, never resample).
- Actual checks: 102 tests passed, including 14 action-world cases. The 45 public
  effect templates have 45 distinct transition tables. Fresh legacy demo and C1
  smoke passed. No world-learning experiment, policy training, final-world sampling
  or performance comparison was run. Tests establish the simulator contract only.
- Next audit: implementation of semantic sampling/fixture exclusions, per-episode
  slot permutations and model updates. Require true-hypothesis retention, genuine
  erasure, fixed-mechanism multi-step predictions, and measured oracle-call budgets
  before planning. Unknown primitives, representation learning, general causal
  discovery, learned planning and transfer remain open research problems.

## 2026-10-23 — Transition beliefs and development sampler

- Scope/source: `action_model.py`, evaluator-only sampler, scripted demo/CLI/CI and
  adjacent simulator transition records/canonical identity; parent `a827f50`.
  Existing arithmetic/graph algorithms, configs and frozen reports are unchanged.
- Reproduced defect: the committed `Transition` dataclass accepted an illegal action
  and mixed before/after catalogs. Re-executed the prior source to reproduce, then
  confirmed the new validator rejects the same record. Regressions cover bad actions,
  mixed catalogs and malformed model input before observation/counter mutation.
- Rollout counterexample: after observing 100 -> 000 for an action, six consistent
  supplied hypotheses remain (guarded/unconditional set-zero or toggle). Two repeats
  from 000 must end at 000 under every fixed hypothesis. Independent-step resampling
  incorrectly assigns 25% to 100. The new joint predictor returns probability one
  for 000 and counts 12 hypothetical effect evaluations; no oracle call is needed.
- Sampling: enumerate unique behavioral identities, not syntax triples/renamings;
  action-permutation symmetry permits combinations with replacement during enumeration.
  Population has 2744 identities, with 1934/410/400 development/validation/final after
  named-fixture exclusions. Development draws match uniform sampling of this unique
  sorted bank. Independent slot permutations change roles while preserving names,
  initial layout and identity. Execution rejects reserved and forged identities.
  Cached public tables match the prior uncached canonical hash on regression fixtures.
- Limits: the model's independent uniform labeled-effect prior is not the sampler's
  uniform canonical-world population. Report model-relative uncertainty; no population
  calibration claim. Vocabulary, filtering and rollout inference are supplied. The
  core learns beliefs from observations, not parameters, primitives or abstractions.
  Malformed structure raises; valid but impossible dynamics give model-class failure.
- Actual checks: 111 tests passed. True effects remain in the candidate sets after
  12 real transitions in each of eight development fixtures. Reset clears evidence,
  posterior and work counters and restores prior predictions; the pure core imports
  no Torch/Transformers/language/sampler. Fresh legacy demo and five-action scripted
  action-model demo passed. The latter leaves 27/6/3 candidates and ambiguity; its
  probe is not oracle-scored, its actions are scripted, and it is not the 80-world study.
  No validation/final mechanism was executed and no performance comparison was run.
- Next: controlled investigation, proper joint prediction scoring and common-unseen
  panel coverage, followed by supplied goal planning. Freeze source/config before
  measurement, track real versus simulated costs, keep all controls, and test that
  evaluator targets and reachability cannot influence choices. Transfer, learned
  search benefit, calibrated uncertainty and acquired abstractions remain open.
