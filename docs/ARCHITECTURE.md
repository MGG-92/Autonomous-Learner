# Architecture and experimental boundaries

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

References: Sutton, Precup & Singh (1999), options,
https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf;
Lewis et al. (2020), retrieval-augmented generation,
https://arxiv.org/abs/2005.11401.
