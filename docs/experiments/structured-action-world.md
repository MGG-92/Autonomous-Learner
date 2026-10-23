# Structured action-world bridge: development contract v1

Status: protocol and simulator contract only. No model learner, investigation policy,
planner, experiment runner, or performance result exists for this environment yet.
The values in `configs/action-world-development.json` are proposed development
settings, not measurements. Preserve C1/C2 results and their negative conclusions.

## Research question and supplied assumptions

Can experience with state-changing actions improve a model's joint predictions and
the success of short plans, compared with its unconditioned model and simple
investigation controls? This moves beyond isolated input/output rules by requiring
state, action ordering, conditional interactions, and consequences. A successful
result would still be bounded learning under supplied representations and algorithms.

Use three objects with one binary property each, three actions, and eight fully
observable states. Public observations contain only opaque object/action IDs and
the ordered property bits. All three actions are available in every state and world;
the legal-action catalog never reveals hidden feasibility, target objects or effects.
Object positions are public representation slots, not semantic roles. Naming seeds
must be sampled independently of hidden mechanisms. There is no language model,
perception front end, partial observability, reward hint, or source-reliability claim.

Each action has an unknown deterministic effect: set one object's bit to zero,
set it to one, or toggle it. The effect may be unconditional or require another
object's pre-action bit to be zero/one. Failed guards leave state unchanged and
still consume an interaction. At most one property changes per action; there is no
automatic cascading. Actions interact because one action can enable another's guard.
These operations, arity limits and stationarity are supplied priors. There are 45
behaviorally distinct effect hypotheses per action; discovering them is not claimed.

## Boundary and sequential experience

`action_world.py` separates immutable `Catalog`, `Observation` and `Transition`
records from the evaluator-owned `ActionWorld` instance and its true `Effect`s.
The future core receives only these public records, public budgets, and a reply
to the action it chose. Do not pass the world instance, true effect objects,
mechanism identity, seed used for mechanism selection, evaluator reachability,
probe targets, or researcher traces into learner state or policy features.
Python API separation is not a security sandbox against arbitrary hostile code.

Exploration starts from one independently sampled state and permits 12 sequential
actions, with no agent-accessible reset, arbitrary state preparation, or hypothetical
oracle calls. Repeated observations of the current state are free and contain no
new hidden facts. Unknown actions and exhausted budgets fail before changing state;
valid no-op actions consume budget. Absorbing states and incomplete identification
are legitimate outcomes. Do not silently reset the world to make learning succeed.

For the first learner, enumerate public effect hypotheses and filter them only by
actual before/action/after observations. Contradictions must produce explicit
model-class failure. Retain dependencies needed to predict entire next states and
action sequences. In a hypothetical rollout the same unknown mechanism stays fixed
across all steps: resampling an action effect independently at each step would
invent a different environment. A future regression must catch this error.
This is belief updating, not parameter optimization or acquired primitives.

## Worlds, names, and transfer claims

Before sampling episodes, enumerate effect triples, quotient by complete transition
behavior, and canonicalize object and action permutations. `mechanism_identity`
implements that canonical identity. Uniformly sample distinct identities, not raw
syntax triples or seeds; multiplicities/automorphisms must not weight the sample.
After selecting an identity, independently permute object and action slots for
the episode before assigning public names. Otherwise canonical representatives
could give positions stable causal roles; changing only ID strings is insufficient.
Use a separate permutation seed range, never hidden-effect-dependent names.
Hash buckets from `int(identity[:8], 16) % 100`: development below 70, validation
70 through 84, future final 85 through 99. Named software fixtures (the three-action
guard chain and three identical unconditional set-zero effects) and all their
renamings are development-only regardless of hash. Implement this exclusion before
any validation/final selection. No sampling/split runner is implemented in this increment.

First implement and measure only 80 development identities using selection seed
611. Seed ranges are independently assigned by purpose in the config. No final
phase or final measurement is authorized by this development protocol. A separate
source/config freeze and review is required before any later final evaluation.
Do not substitute a renamed previously opened mechanism for a fresh mechanism.

New IDs/layouts test invariance; new canonical mechanisms test broader recombination
within the supplied family. Neither establishes new primitive/family transfer.
The first per-world learner starts fresh, so it cannot demonstrate transfer of
acquired abstractions between worlds. Later C3 must compare retained learned
libraries/parameters with from-scratch and library-removal controls on new mechanisms.

## Committed predictions and goal tests (future runner)

After exploration, freeze each learner's evidence/model for scoring. Predictions
must be computed before reading evaluator outcomes; probe replies must never update
the model. Use the complete 24 state/action panel and 16 two-action probes sampled
from the 72 possible state/action/action triples using an independent panel seed.
The core may simulate hypotheses at these states but cannot query their true effects.
Report proper joint next-state Brier and log scores over the eight states, plus
accuracy. Per-bit accuracy alone can conceal joint/sequence errors.

Panel membership is fixed independently of mechanisms and exploration. Some panel
tuples may have appeared in actual experience. Tag those explicitly. For paired
unseen comparisons, use the panel tuples absent from every compared policy's
experience in that world; compute this intersection outside all learners, after
their exploration. Report its size/coverage and conditional scope, as well as full
panel results. If empty, report unavailable; never resample or call it generalization.
A sequence not previously executed may still consist of familiar single transitions.
This distinction must be visible in results. World splits remain the stronger
boundary; this conditional panel is not an independent final test distribution.

Goal tests start a fresh evaluator-controlled episode at the same public initial
state for every policy, preserving each frozen learned model. This episode boundary
is not a free reset available during exploration. Sample the desired full state
uniformly from the seven states different from the goal-start state, independently
of true reachability. Do not rejection-sample achievable goals or provide a gold path.
The evaluator may report true three-step reachability afterward, never as input.

Commit an open-loop action sequence of length at most three before any goal-phase
feedback. A supplied exhaustive planner can maximize predicted goal probability
over fixed model hypotheses; break equal probabilities by shorter plan then public
seeded tie-breaking. Execute the committed plan without online model updates or
replanning. Report goal success on all sampled tasks, actions used and model-relative
predicted success; do not count oracle-conditioned task selection as intelligence.
Later online planning is a separate experiment. The planner is supplied even when
it succeeds through an experience-learned transition model.

## Controls, costs, and review before execution

Use the same hidden worlds, exploration starts, goal starts/goals, probe panels,
and interaction ceilings across policies. Keep exploration and planning contributions
separate: random exploration with updated model and supplied planner; fixed cyclic
exploration with updated model and planner; an information-gain heuristic over
the current state's possible transitions with updated model and planner; the
untrained prior/no-memory model with that planner; and a random goal plan control.
Untrained and reset/no-memory are equivalent when there are no trained parameters;
report the equivalence instead of claiming independent evidence. For a later
trainable controller retain its exact random initialization as an additional control.

Record actual actions including no-ops, candidate effect evaluations, joint rollout
work, and wall time per policy, plus any future optimization cost. Count hypothetical
model predictions separately from real environment interactions. Matched interaction
ceilings do not imply matched computation. Start with three policy RNG seeds
7/42/97, shared worlds; report individual seeds and aggregate within worlds before
any world-level intervals. Do not treat seed repeats or probe rows as independent
worlds. No arbitrary success gate or general-reasoning claim is predeclared.

Before running: test leakage/renaming invariance, noncommuting actions, hard budgets,
true-hypothesis retention, joint rollout consistency, no-memory erasure, and oracle
call counts. Review at least the model and evaluator boundaries. Persist source
hash, dirty status, all settings and seeds, actual denominators, unreachable-goal
counts, ambiguity and negative outcomes. Publish reviewed synthetic summaries only.
Learner implementation, all controls and measurement remain unchecked roadmap items.
