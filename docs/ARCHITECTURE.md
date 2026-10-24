# Architecture and experimental boundaries

## Active native embedding experiment

The user redirected development to `neural-volume-lab/`. Its desktop application
uses native PySide6 QtWidgets and QPainter perspective drawing, not a web interface.
A frozen, pinned input-embedding table maps actual token IDs to 576-component vectors;
consecutive triples become vertices joined in order. One fixed table-wide scale
preserves relative component magnitudes across tokens. Token playback and camera
changes never run the model or alter vectors. Response references are separate
visualizations; explicit training and free predictions remain separate.

Offline export converts cached bfloat16 values exactly into read-only float32 assets,
with source revision and checksums. The packaged runtime uses local NumPy/tokenizers
and Qt. A native QProcess bridge starts this checkout's existing CPU Python runtime
on request; Torch/Transformers are not duplicated into the GUI bundle. The worker
loads a sparse recurrent learner and lazily loads the pinned professor offline.
The UI identifies its pretrained source; raw component geometry is not semantic
3D distance. Pretrained embeddings are authorized for this separate experiment.
The earlier restrictions below continue to describe the preserved controlled studies,
not a prohibition on the new user-requested embedding viewer.

The learned network has 25³ units, 12 directed inputs per unit, random initial
weights and zero initial activity. Spatial vertices become binary nearest-site
pulses; token order is temporal, but vertex order and quantization detail are lost.
Synchronous leaky ReLU recurrence has a projected row-L1 bound of 0.7 and an activity
ceiling of 4. It contracts without input and cannot spontaneously leave zero.
Readout projection and 16 ordered token slots plus EOS are learned with cross entropy,
Adam and eight-pair replay. Targets are labels, not clamped predicted states; no
spatial reconstruction objective is implemented. No message-to-reply lookup occurs
at prediction. The finite target-token vocabulary is supplied readout support.

User targets, unverified frozen-professor proposals, live neural activity, reference
geometry and free network output have distinct UI roles. Manual lab targets require
Train pair; the newer chat path below has its own explicit correction and enabled
coaching workflow. Silence preserves parameters; Reset learning erases
them. Rest training uses perturbed states plus task rehearsal. All interaction and
learning is RAM-only; subprocesses close with the app. There is no service, model
download or background teacher loop. New application-controlled caches/builds stay
under ignored runs on the checkout drive, reusing existing weights. Packaged neural
compute requires the checkout's .venv. Two-pair fitting and numerical tests establish
software behavior, not learned reasoning; a frozen recurrent core also fits the pairs.

### Chat and immediate feedback (user-directed extension)

The default native view is a chat beside the volume. Start brain initializes silent
activity and starts an adjustable clock (default five seconds). Each tick supplies
exactly the next queued token pulse and applies one recurrent update; after input,
ticks are undriven. The visible attempted reply is sampled from the actual current
state and learned slot readout. It never calls the professor for its own output.
Full queued input is not supplied ahead of the current token; credit/teaching uses
the received prefix. Wall-clock pacing can be delayed by compute; requests serialize.

Human +1/−1 feedback uses a clipped immediate-reward token-policy surrogate and
SGD step 0.05 × impact (default 5), gradient-norm cap 1, one update per source/reply.
Behavior log probabilities permit clipped reuse after teaching; this is not full
PPO, a value model, or long-horizon RL. A live credit snapshot contains the previous
state and pulse, with earlier state detached. Only the current transition/readout
receives temporal credit. The last eight reply snapshots actually expire; IDs bind
ratings and edits to the selected reply rather than the newest reply by accident.

Double-clicking a reply pauses ticking and opens its actual sampled text. Teaching
the edit uses token cross entropy on that captured context, 60 Adam updates and
eight-example replay. It does not display the edited text as if the network said it.
Human examples replace earlier professor examples for the same received input and
block professor imitation overrides while retained. Ratings never fabricate targets.

Optional professor coaching (on by default) runs once after a sentence is received,
not at every idle tick or as a background job. Frozen next-token Yes/No logits yield
an unverified reward (+1 if relative Yes support ≥0.65, −1 if ≤0.35, otherwise zero).
This is uncalibrated and teacher correctness is not validated. Professor reward weight
is 0.25, impact 1. A first/new-word example can teach 20 supervised updates; otherwise
coaching gives reward only. This hybrid bootstraps vocabulary without pretending
scalar rewards introduce unseen words. All data and weights remain session-only.

## Preserved cognitive-core objective

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

## Action-world bridge: development prediction comparison implemented

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
contract in `docs/experiments/structured-action-world.md`. Prediction-only development
results are in `reports/action-prediction-v1/README.md`. A separate goal evaluator
is now implemented under `docs/experiments/action-goals.md`.

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
The scripted `action-model-demo` records changed predictions and exact reset.
`action_investigation.py` supplies information gain, random and cyclic controls using
only current public state and the model. `action_prediction.py` commits all policy/seed
predictions before target access and scores full/shared-unseen one/two-step panels.
The first development comparison supports improved prediction from experience, with
bounded supplied inference and myopic investigation. Goal planning, parameter
learning, new abstractions, calibrated uncertainty and transfer remain pending.

`action_planning.py` now provides the next supplied primitive: exhaustive open-loop
search of sequences up to three actions using only the acquired model and public
start/goal. It ranks predicted success, prefers shorter tied plans and uses a public
RNG for remaining ties; it reports simulated work without querying an environment.
Software fixtures verify planning from experienced transitions. `action_goals.py`
adds independent goal/start sampling, IG/random/fixed investigation, prior-only and
random-plan controls. It saves all policies/seeds' commitments before goal feedback
or evaluator-only reachability. Execution receives no model and cannot replan or
update evidence. The first clean-source development result is in
`reports/action-goals-v1/README.md`: only 10/80 sampled goals are reachable, and IG
versus prior has an interval crossing zero. No learned planner, self-model or
mechanism-family transfer is demonstrated.

## Functional self-modeling: proposed extension, not implemented

The research objective includes learning about the agent's own reliability and the
consequences of memory, strategy and compute choices. This can use the same learning
principles as world modeling: predict from accessible context/operation, observe
actual feedback, update and test interventions. Reusing principles does not imply
reusing the finite guarded-effect grammar or requiring a separate introspection module.
Begin after the goal baseline with a small empirical success/error predictor; compare
it with simple calibration and equally informed controllers before adding complexity.

Current posteriors, accessible evidence, reset controls and strategy/cost records are
potential signals, not an existing self-model. New mechanisms are needed for learned
self-related predictions, agent-specific attribution and adaptive metacognitive control.
Candidate counts/entropies describe uncertainty within the supplied class; they do not
establish knowledge of blind spots or calibrated reliability. Self-model learning
and strategy/parameter optimization must be reported separately.

Future agent-visible cognitive events must have an explicit bounded interface,
distinct from researcher traces and audit logs. Only actually accessible evidence,
chosen operations, declared resource observations and ordinary delayed feedback
may enter it. Hidden mechanisms, held-out panel answers, evaluator reachability,
intervention flags and evicted task contents stay unavailable. Cost the interface,
self-model state and control actions; it must not become an unlimited auxiliary memory.

World and self models may share representations. Testing their functional separation
requires controlled task/agent variations, task-only and no-self-model controls,
monitor-only and shuffled-state ablations, and intervention benefits after overhead.
Language can later render committed self-related claims but cannot supply them.
The staged proposal in `docs/SELF_MODELING.md` makes no subjective-consciousness claim.

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

The planned Interactive Agent Playground is a development-only CLI adapter to the
actual action-world model, investigation and goal-planning execution path. P1 follows
the first scored goal baseline; no playground command or interactive action-world
session exists yet. See `docs/PLAYGROUND.md` for scope and acceptance gates. Structured
goals and queries precede natural-language input. Versioned operation events support
step/run and future graphical views without rerunning decisions during inspection.
Observer history stays outside learner inputs; sandbox mechanisms and interactions
cannot enter reserved evaluation. Additional user-requested predictions are explicit
work, distinct from reading captured telemetry. Later memory, abstraction and
self-model views expose implemented mechanisms, not simulated introspection.

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
