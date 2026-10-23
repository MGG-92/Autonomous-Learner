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

Next: learn search/investigation strategies against this explicit baseline, then
induce reusable abstractions. A future language adapter proposes typed observations
and renders authorized structured claims; its parsing/fidelity must be evaluated
separately, and its wording must not automatically become new cognitive evidence.
See `docs/experiments/internal-rule-discovery.md` and C1–C4 in `docs/PLAN.md`.

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
