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

## 2026-10-23 — First action-world prediction comparison

- Scope/source: investigation controls, prediction evaluator, CLI/CI and adjacent
  transition-model/sampler boundaries. Clean source/config/protocol freeze
  `b94d48acb3880e88e244f093cf196ef073cfef3f` before the 80-world development run.
  No implementation or hyperparameter change followed the measurement.
- Boundary checks: policy takes only the current observation, model and public RNG.
  The sole real-action callback must match the chosen action and pre-state before
  observation learning. A regression verifies all controls/seeds commit every panel
  prediction before any evaluator target access. All 720 actual investigation runs
  consume 12 actions (8640 total), counting no-ops; 240 reset records reuse IG traces.
- Conditional-panel critique: using own-trace unseen cases would compare different
  questions and can leak another policy/seed's familiar cases into an unseen claim.
  The implementation excludes the full experience union across all compared runs.
  Regressions reconstruct that complement, keep empty-panel denominators explicit,
  and preserve outputs on rerun. This measured run had no empty shared panels.
  Coverage is 17.325/24 one-step and 12.075/16 two-step cases per world; 47/966
  unseen two-action probes contain single transitions observed somewhere in the union.
- Counterexample to broad interpretation: the reset prior always predicts no change
  as its argmax, attaining 1190/1920 one-step and 722/1280 two-step correct outcomes.
  IG also experiences 9.0625 no-ops/12. This post-hoc saved-target check shows why
  accuracy alone can overstate understanding. No new policy or world rerun was used.
- Evidence: IG full accuracy 75.31%/71.80%, random 73.63%/69.97%, reset 61.98%/56.41%.
  Shared-unseen Brier reductions versus random are +0.0173/+0.0223 and versus reset
  +0.1415/+0.1365; exploratory world-bootstrap intervals are positive. Two-step
  unseen versus fixed +0.0105 has interval [-0.0055, 0.0289], crossing zero; no
  universal heuristic benefit or acquired reasoning strategy is established.
- Actual checks: 121 tests, fresh legacy/model demos and scoring smoke passed.
  Verified all 3840 scores against saved joint probabilities/targets and shared
  unseen sets; checked every aggregate, paired difference and bootstrap interval.
  Independently reconstructed effect-evaluation counts for all 960 cost records
  from public traces without executing worlds. No new correctness defect was
  reproduced in the adjacent transition-model path; existing limitations remain.
- Cost/limits: IG exploration 1043.1 effect evaluations versus random/fixed
  293.6/291.3; compute measured, not matched. Priors, inference, effects and
  investigation remain supplied. Reset/no-memory/untrained are equivalent here,
  not independent controls. Model priors differ from population weights. All
  measured worlds are development; no validation/final, goals or transfer tested.
- Next: supplied open-loop goal planning, independent goals, prior/random-plan
  controls and unreachable-goal accounting outside learner inputs. Freeze that
  scope before measuring; preserve this report and all earlier negative findings.

## 2026-10-23 — Functional self-modeling roadmap and planning primitive

- Scope/source: long-term objective, roadmap, architecture and new supplied planner;
  adjacent world-model probabilities, memory/audit boundaries and existing reports.
  Parent `77592da`; frozen prediction experiments, configs and active policies unchanged.
- Critical distinction: confidence/candidate counts and stored history are available
  signals, not learned self-understanding. A successful task-only forecaster can
  mimic reliability awareness; a strategy selected for harder tasks can have a
  higher failure rate without being worse. Proposed C5 tests control these alternatives
  using matched task/agent interventions, calibration baselines and behavioral ablations.
- Information-budget risk: feeding researcher logs or forgotten task contents into
  a monitor would bypass the established learning boundary. No such interface was
  implemented. The new contract requires declared bounded internal events and normal
  feedback, excludes evaluator targets/condition flags, and charges extra resources.
- Conceptual limits: an implicit history-based controller may learn equivalent
  adaptation without a separately named self-model. A separate module must earn
  its complexity; equal results do not prove the baseline lacks implicit metacognition.
  Forecast improvement without control benefit is monitoring only. Functional gains
  do not establish subjective consciousness. Primary author research and the
  Fleming/Lau measurement review ground distinctions, not project capability claims.
- Continued implementation: a supplied exhaustive planner consumes only acquired
  dynamics and public start/goal, evaluates sequences up to length three, commits
  before execution, and records simulated work. No actual environment callback or
  self-model enters planning. Predicted success is task-model-relative, not learned
  empirical self-reliability. This primitive is not integrated into an agent policy
  or scored goal study yet; no policy or metacognition measurement was run.
- Actual checks: 127 tests passed, six new planning cases. The named development
  chain is planned from three observed transitions and then executed without changing
  learned evidence. Ties prefer shorter plans with seeded choice; zero/bad budgets,
  model failure and effect counts are explicit. Fresh legacy and action-model demos
  passed. No new correctness defect was reproduced in the adjacent belief core.
  Independent local roadmap formatting was preserved and excluded from the commit.
- Next: scored development goal baseline with independent goals, prior/random-plan
  controls and source/config freeze. Then specify the C5a reliability pilot and
  its feedback schema. All C5 milestones remain unchecked, as do goal-performance,
  learned strategy, new abstractions, calibration and transfer claims.

## 2026-10-23 — Interactive Agent Playground planning

- Scope: roadmap and observer contract; inspected the real investigation loop,
  planning primitive, architecture, CLI and adjacent historical viewer requirements.
  No runtime behavior changed and no new runtime defect was reproduced.
- Counterexample: implementing step mode by rerunning action selection could consume
  RNG and simulated work, changing the supposed explanation. P1 requires one resumable
  core execution path, captured actual-operation events and recording/step/run parity.
  Inspection reads recorded telemetry; fresh prediction requests are explicit work.
- Scientific boundary: new seeds or object labels can reproduce reserved mechanisms.
  Manual sessions must use the development partition; later edits/imports need semantic
  identity/exclusion checks. No automatic training ingestion or benchmark promotion.
  Silent rule reveal/model reset would fake adaptation; keep edits deferred and expose
  current fixed-model failure honestly. These are design requirements, not verified
  playground protections: no playground code was added.
- Roadmap conflict resolved: P1 follows scored goal learning, while the retained M2.5
  graphical viewer and its historical delayed-memory dependencies do not block it.
  C5a/C3 research need not wait for graphics, language or interface polish. Existing
  memory facilities are not yet integrated into the action-world path; planned views
  must not imply that they are, or turn observer history into agent memory.
- Actual checks: all 127 tests, fresh legacy and action-model demos, diff whitespace
  checks and publication preflight passed. No policy training or new scientific
  measurements were appropriate.
- Next audit: independent goal/start sampling, prior/random-plan information and cost
  fairness, commitment before feedback and source/config freeze in the scored goal
  runner. Later P1 must test its observer and partition boundaries before launch.

## 2026-10-23 — Goal evaluator source/config freeze

- Scope: `action_goals.py`, planner, investigation and simulator boundary; inspected
  the adjacent prediction runner and preserved its frozen measurements/protocol.
- Concrete risks: choosing only reachable goals gives hidden-mechanism information;
  planning after another seed's outcome can contaminate decisions; feeding execution
  replies back into the model changes an open-loop test into online learning.
  The runner samples goals independently, persists every policy/seed commitment before
  any goal truth, and executes without a model argument. Regressions enforce these
  boundaries, unreachable tasks and genuine reset erasure. No old defect is claimed.
- Fairness: reset and random-plan reuse IG exploration costs and traces; they are
  controls, not additional investigation episodes. Same goal ceilings permit different
  plan lengths. Prior computation is substantial: a supplied software fixture used
  1774710 effect evaluations. Costs are measured, not matched. Random plans use the
  full ceiling with uniform action slots, unlike shortest-tie model plans.
- Analysis: report all goals primarily, reachable subgroups with real denominators,
  world-paired differences after seed averaging, exploratory/unadjusted world bootstrap.
  One goal per development mechanism cannot establish broad goal generalization.
  Model-relative success is not learned self-reliability or empirical calibration.
- Actual checks before freeze: new boundary tests and two-world/one-action CLI smoke
  passed. Smoke goals happened to be unreachable; no synthetic success was substituted.
  All 139 tests and fresh legacy/action-model demos passed. Full measurement
  requires a committed source/config and clean tree; no reserved worlds will be opened.
- Next: reconstruct the measured report from saved commitments, transitions and costs;
  check source identity, same tasks, exact action counts, reachability denominators and
  negative outcomes before publication. Goal results do not prove learned planning.
