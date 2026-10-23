# Architecture and experimental boundaries

## Clarified objective and active direction

The agent should acquire and revise an internal predictive/explanatory model from
experience, choose informative experiments, and commit what it intends to communicate
before optional linguistic rendering. The controlled core must run without an LLM,
pretrained text embeddings, language-generated rules, or language-generated answers.
This separates responsibility; it does not make a learner free of inductive biases.

The first model-learning baseline enumerates a supplied depth-two program language
over a finite 5x5 input domain, with modular arithmetic/min/max primitives. Equivalent
expressions count as one semantic hypothesis. Observations filter a uniform version
space; a supplied information-gain heuristic chooses questions. Predictions are
distributions over outcomes, conditional on this finite model class. No parameter
optimization, primitive discovery, or learned reasoning strategy is claimed.

The environment/evaluator owns the hidden rule and reserved target outputs. The core
receives only the public hypothesis space, legal input pairs, and observations from
queries it actually makes. All policies share the same reserved inputs per world.
Model reset removes both posterior restrictions and raw observations; researcher
traces stay outside learner state. Splits use whole finite-domain behavior, preventing
equivalent expressions from leaking across partitions. Generalization outside this
domain/grammar and across new primitive families remains untested.

The first query-policy optimization against this baseline is implemented in
`query_policy.py` and `query_training.py`: a shared scorer of sorted predictive
masses trained by prediction-reward REINFORCE, with no entropy labels or demonstrations.
Initialization is already strong; the first development comparison does not show
reliable improvement over it. Filtering, features, reward, and grammar remain supplied.
See `reports/learned-investigation-v1/README.md` for the measured limits. Next implement
the specified action-world learner, preserving open search/transfer questions, then
investigate reusable abstractions. A future language adapter proposes typed observations
and renders authorized structured claims; its parsing/fidelity must be evaluated
separately, and its wording must not automatically become new cognitive evidence.
See `docs/experiments/internal-rule-discovery.md` and C1–C4 in `docs/PLAN.md`.

## Action-world bridge: transition beliefs implemented, experiment pending

Finite function discovery is a diagnostic for a broader objective: learning predictive
models of objects, properties, actions, interactions, and consequences, and using
those models to achieve goals. A function score alone cannot demonstrate this.
After a bounded learned-search comparison, introduce a small fully observed discrete
world before extending to partial observation, noisy perception, or language input.

Keep the environment boundary structured: observations, legal actions, and experienced
state/action/next-observation transitions. Hidden transition mechanisms and held-out
outcomes belong to the evaluator. The learner may have an explicit representation
prior, but must acquire which effects hold from experience. Goals specify desired
observable outcomes, not the hidden action sequence or causal rule that achieves them.

Distinguish the learned predictive model, investigation strategy, and goal-directed
planner. Evaluate predictions before feedback and test model interventions/reset;
if planning uses supplied search, report it as supplied search over learned dynamics.
Controlled interventions can test causal predictions within the simulator's declared
assumptions; observational prediction alone does not establish causal identification.
Research traces and evaluator state must remain inaccessible to all three components.

Begin with declared fixed primitives as a baseline, then evaluate acquired reusable
fragments/concepts and their contribution through transfer and removal ablations.
New object labels or layouts test narrower invariances than new mechanisms or unseen
compositions. Separate these claims and split whole worlds accordingly. Learning
without any representation bias is not a requirement; learning beyond a permanently
fixed list of complete answers is the research direction. These are future interfaces
and evaluation requirements, not capabilities of the current implementation.

`action_world.py` now supplies the initial evaluator-owned simulator: three fully
observed binary properties and three actions with unknown guarded set/toggle effects.
Public immutable records expose object/action IDs and observed bits, not true effect
instances or mechanism identities. Legal actions are independent of hidden effects;
no-ops consume budget and there is no agent-accessible reset. This is a software
oracle/interface, not a learned policy. See the v1 development
contract in `docs/experiments/structured-action-world.md`. No study has run yet.

`action_model.py` now filters the public effect hypotheses from experienced
transitions and preserves their joint implications in rollouts of up to three
actions. It receives only a public catalog and observed transition records. Reset
clears candidates, raw observations and model-work counters; inconsistent evidence
causes explicit model-class failure. The inference algorithm and effect vocabulary
are supplied; learning is posterior conditioning, not parameter optimization.
Independent uniform labeled-effect priors differ from the evaluator's uniform
canonical-world population, so probabilities are explicitly model-relative.

A later supplied planner must use the acquired model and commit a short sequence
before goal feedback. Canonical
world identities quotient object/action permutations for future split integrity;
episode layouts must independently shuffle the slots to avoid canonical role cues.
`action_sampling.py` now samples development identities uniformly without replacement
and independently shuffles episode slots before naming. It executes only verified
development mechanisms, keeping named software fixtures development-only. Public
class enumeration yields 2744 identities (1934/410/400 development/validation/final).
It does not execute reserved worlds. Cached public transition tables preserve the
original identity algorithm while making complete-class enumeration CPU-friendly.
The scripted `action-model-demo` records changed predictions and exact reset;
investigation policies, scoring, planning, parameter learning and transfer remain pending.

## Earlier controlled graph track (supporting baseline)

The controlled track has no language model. A generated hidden world belongs only
to the synthetic teacher and evaluator. The agent sees a catalog of legal questions,
its goal, teacher observations, and its own memory. A neural candidate scorer selects
ASK, ANSWER, ABSTAIN, or WAIT. Answer content comes from supplied graph inference.

An entity belongs to a class, classes may have parents, and a root class has an
explicit positive or negative capability. Graph inheritance is monotonic; conflicting
supported polarities produce CONFLICT rather than latest-write-wins. Missing evidence
is UNKNOWN, not false. Rule inference is supplied, not learned.

Memory stores evidence with provenance, timestamps, and stable IDs in SQLite. Its
bounded capacity really deletes old accessible evidence. Inspection and audit traces
are for researchers and are never fed back into the agent. The initial admission and
eviction strategy is a fixed FIFO baseline. A later milestone will learn it.

The policy sees candidate features derived ONLY from observed graph reachability,
query history, goal overlap, answer availability, and remaining budget. Random names
do not have stable role prefixes. Supervised expert labels can inspect hidden worlds
in training; that information never enters evaluation policy inputs.

Parameter learning uses expert imitation followed by REINFORCE on separate generated
training worlds. Evaluation compares the frozen learned checkpoint against random,
untrained, fixed heuristic, and memory-disabled variants. The same task distribution
and action budget are used. Randomized worlds test new content, not acquisition of
grammar or unconstrained reasoning. Match a good heuristic before making stronger claims.

The later language track turns a structured claim and evidence references into text.
Giving evidence to an LM can outsource reasoning; controlled-track results remain the
reference for attributing learning to this architecture. A token reranker is optional
and must earn its complexity through ablation. Syntax/semantics are not cleanly separated
inside a pretrained LM.

The newer delayed-question memory track withholds the specific question until after
a shuffled stream and quiet delay. A separate neural policy jointly encodes visible
evidence, scores retention, and chooses WAIT/CONSOLIDATE under hard write/storage
limits. Its possible query distribution/vocabulary and horizon remain public. It is
bootstrapped from a supplied graph heuristic and optionally refined by terminal-reward
policy gradients. See the versioned protocol and results; this is not learned logic
or joint training with the original question-selection controller.

The planned M2.5 viewer is a read-only observer of versioned memory snapshots and
activity events from the persistent loop. A minimal local 3D graph and timeline
will support saved-session replay before live observation. Display current bounded
memory separately from historical replay; researcher history must never become
agent-accessible memory. Stable evidence IDs link selected graph edges to actual
provenance and inference, and the viewer must not expose teacher-hidden facts.
CLI operation and scientific evaluation remain independent of visualization.
Record operation-level events at run speed and replay without invoking the policy.
The explanatory component view connects observations, the two memory stores,
controller inputs/scores/actions, and supplied inference evidence. Its activity
indicators describe actual recorded execution, not simulated thoughts. Optional
neural activation details must be captured from actual computations and clearly
labeled; scores and activation magnitude are not calibrated confidence or causal
explanations. Recording must preserve decisions and keep observer data inaccessible
to the agent; measure its overhead independently of replay speed.
See `docs/PLAN.md` for dependencies and acceptance criteria.

Persistent question-loop sessions use schema-v2 identities for task/budgets, feature
names, controller weight digests, memory settings/content digests, and teacher type.
An ASK reserves its budget and records pending intent before contacting the teacher.
Incomplete ticks or inconsistent state stop further actions rather than replaying an
uncertain call. SQLite stores each observation/eviction atomically, but SQLite and
session JSON remain separate transactions. Legacy sessions are preserved and refused
by the new schema; automatic crash recovery and concurrent writers are unsupported.
Observer logs never repair or augment policy memory. See `docs/CRITIC_AUDIT.md`.

References: Sutton, Precup & Singh (1999), options,
https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf;
Lewis et al. (2020), retrieval-augmented generation,
https://arxiv.org/abs/2005.11401.
