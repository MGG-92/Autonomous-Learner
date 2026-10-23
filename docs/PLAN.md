# Implementation plan

## Current priority — independently learned internal models

The user's clarified objective is a cognitive core that discovers and manipulates
internal models from experience, with an optional pretrained LLM for communication.
The earlier graph experiments below are supporting baselines: they learned control
and retention around supplied inference, not rule discovery. Follow C1–C4 first;
memory infrastructure, token control, and the viewer do not establish this objective.

The destination is reasoning about worlds: objects and properties, actions and
interactions, consequences, and goals. Mathematical functions provide an objectively
scored starting point, not an endpoint. Keep C1 frozen as a diagnostic and preserve
its negative findings. C2 remains next; take a bounded learned-search increment,
then the structured-world bridge below before prolonged function-specific tuning.
Neither a fixed hypothesis class nor a supplied planner is the intended end state.

### C1 — Bounded unfamiliar-rule discovery baseline
- [x] Generate compositional expression hypotheses independently of the oracle.
- [x] Update competing internal models from observed input/output pairs, without an LLM.
- [x] Choose experiments by supplied information gain; predict reserved inputs.
- [x] Compare random, fixed/untrained query control, no-memory, and model-reset controls.
- [x] Split semantic functions rather than renamed expressions; test model failure honestly.
- [x] Freeze and evaluate the first protocol; publish results and causal/uncertainty limits.

This baseline learns a rule within a supplied finite language by belief updating.
It does not learn its primitives, search algorithm, or reasoning strategy.

### C2 — Learn investigation/search strategies
- [x] Train a first query policy on separate development functions without language priors
  or query demonstrations; preserve its mixed/negative initialization comparison.
- [ ] Compare against the C1 information-gain/program-search baseline under matched costs.
- [ ] Hold out functions and compositions; reserve new final tasks before evaluation.
- [ ] Ablate internal simulation/model updating and test behavior under counterexamples.
- [ ] State the proposed transferable capability and measure observation and compute
  costs, including training amortization, before selecting a new final protocol.

First bounded development comparison is complete; see
`reports/learned-investigation-v1/README.md`. Training benefit over initialization
is not reliable; information gain wins on Brier in all seeds. Cost counters and
model-access/reset controls are implemented, but matched compute, family transfer,
and reliable learned-strategy benefit remain open. Do not label C2 complete. Next
specify the minimal structured action-world development protocol below rather than
extensively tuning this arithmetic comparison or opening a new arithmetic final set.

### C2 bridge — Small structured action worlds

Plan this after the first bounded C2 comparison and before extensive arithmetic
optimization. Start with a tiny, fully observed discrete simulator and structured
object/property observations, legal actions, and observed transitions. No graphics,
natural-language parsing, or pretrained representations are prerequisites.

- [x] Specify the v1 development protocol for unknown action effects and interactions,
  with an executable simulator contract; keep mechanisms and probe targets evaluator-only.
- [x] Implement development-only unique-mechanism sampling, independent slot permutations,
  experience-conditioned effect beliefs, reset erasure and consistent joint rollouts.
  These are software checks, not a measured goal/transfer result.
- [x] Compare supplied investigation controls and scored joint predictions with
  experience-conditioned beliefs and reset/prior controls. Preserve shared unseen
  coverage and within-class belief updates; structural/noisy revision is still open.
- [ ] Test unseen interventions and action sequences, then goal achievement using
  learned predictions. Label any supplied search/planning algorithm explicitly.
- [ ] Compare random, untrained, heuristic, no-memory/model-reset, and from-scratch
  controls under matched interaction and declared compute budgets.
- [ ] Separate renamed-object/layout generalization from transfer across mechanisms
  or new compositions. Hold out whole worlds/mechanisms, not just transitions from
  training worlds; freeze selection before opening a fresh final split.

Success here would establish bounded action-world modeling under declared simulator
assumptions. It would not establish unrestricted causal discovery or general reasoning.
Choose the minimal environment and model family through development experiments;
do not expose gold causal rules to make the learner appear more general.

Contract: `docs/experiments/structured-action-world.md`; proposed settings:
`configs/action-world-development.json` (full goal study not runnable yet), with
the measured prediction-only subset in `configs/action-prediction.json`. The simulator
has three binary-property objects and three unknown guarded effects; software tests
cover sequential interactions, action budgets, public-record boundaries and semantic
identity under all object/action renamings. The sampler and transition-belief core
are now implemented, with a scripted `action-model-demo`. The core learns by belief
updating within supplied templates, without parameter training. The first 80-world
development prediction comparison is recorded in `reports/action-prediction-v1/README.md`.
No validation/final, goal or transfer result is established. Next implement supplied
open-loop goal planning with independent goals and prior/random-plan controls.

### C3 — Learn reusable abstractions
- [ ] Induce reusable program fragments/concepts from experience rather than adding gold rules.
- [ ] Measure transfer and search savings on new tasks; test representation limitations.
- [ ] Extend revision/noise/model-class uncertainty beyond deterministic in-class filtering.
- [ ] Test learned abstractions in the structured action worlds: compare transfer
  against learning from scratch and removing the acquired abstraction library.
  Measure interaction/search savings as well as prediction and goal success.
- [ ] Document the supplied representation and composition operators separately
  from acquired concepts. Test new mechanism compositions and failures outside
  the current vocabulary; do not count object renaming as abstraction discovery.

### C4 — Communicate established internal content
- [ ] Define structured prediction/hypothesis/question messages and deterministic renderers.
- [ ] Evaluate an optional read-only LLM renderer separately for semantic fidelity.
- [ ] Evaluate language input parsing separately; no unchecked parser hypotheses become beliefs.
- [ ] Build the requested viewer around actual model changes/computations after C1 evidence.

Token-level control remains a deferred optional comparison, contingent on measured
benefit over structured communication. Preserve old experiments and negative reports.

## M1 — Controlled learning experiment
- [x] Synthetic oracle with randomized names, graph relations, explicit negatives.
- [x] Persistent bounded evidence memory and traceable multi-step inference.
- [x] Agent with structured actions and no access to hidden world state.
- [x] Trainable question/action controller, expert bootstrap, policy-gradient fine-tuning.
- [x] Held-out comparisons, memory ablation, checkpoints, learning curves, CLI demo.
- [x] Tests, CI, Git history, public remote repository, continuation documentation.

## M2 — Learned memory management
- [x] Separate bounded episodic and belief stores (original experiment used fixed timing).
- [x] Task distribution with delayed reuse and controlled distractors; broaden later.
- [x] Learned retention scorer versus FIFO/untrained/heuristic with hard budgets.
- [x] Integration of frozen learned retention and question controllers in tick loop.
- [x] Learned WAIT/CONSOLIDATE and retention scores, without the specific future question.
- [x] Shuffled streams, expiring episodic evidence, matched write/storage budgets, predefined gate.
- [x] Three-seed delayed-memory test, imitation-only ablation, storage-only upper bound.
- [x] Frozen-policy robustness evaluation across changed budgets, horizons, order, and observation gaps; failures documented.
- [ ] Integrate the delayed-memory controller into the persistent investigation loop.
- [ ] Joint training with question selection.
- [ ] Learn retrieval; broaden timing/resource training and evaluate on new held-out worlds and retention delays.
- [ ] Contradiction handling, source reliability, revision, retention curves.

## M2.5 — Minimal 3D memory and activity viewer

Deferred by the clarified cognitive-core priority above. Retain this requested
design, but do not use it as a prerequisite for demonstrating learned internal models.

User-requested observability milestone. Build after the changing-budget/horizon
robustness increment and integration of the delayed-memory controller into the
persistent investigation loop, before expanding M3 autonomous investigation. It
does not need to wait for every M2 research question to be solved. Keep CLI and
evaluation runs independent of the viewer.

Purpose: teach the user what the agent's "brain" is doing as it works. The graph is
one view of a recorded execution, not the whole explanation. Use simple labels and
progressive detail to explain observed information flow between observations,
episodic memory, belief memory, the learned controller, and supplied inference.
Record at execution speed and replay slowly; playback must not rerun the policy.

- [ ] Define versioned, read-only memory snapshots and activity events from the
  actual loop, with stable evidence IDs, tick ordering, and restart behavior.
- [ ] Provide a local viewer launched with one documented command, starting with a
  saved synthetic session and replay; add live observation after replay is reliable.
- [ ] Record ordered steps within each tick, not just final graph states: inputs
  observed, available candidate actions/facts, actual controller scores and chosen
  action, budget changes, memory changes, and any inference evidence IDs. Include
  model/checkpoint identity and trace-schema version so replay is interpretable.
- [ ] Add a compact component view beside the 3D memory graph. Activate components
  as their recorded operations occur; show candidate scores on demand and explain
  each operation in plain language. Clearly distinguish learned policy decisions,
  programmed inference, and stored evidence. Missing telemetry is labeled unknown.
- [ ] Offer an optional advanced view of actual recorded neural activations only
  when useful. Label their numerical meaning and scale; activation magnitude or an
  action score is not confidence, a thought, or proof of causal importance. Start
  with component activity and scores before adding individual-neuron displays.
- [ ] Render a minimal, visually appealing 3D graph: entities/classes as nodes,
  labeled fact edges, restrained colors, subtle motion, legible labels on selection,
  orbit/zoom/reset controls, and a reduced-motion option. Avoid decorative effects.
- [ ] Distinguish episodic and belief evidence; selecting an edge shows its polarity,
  provenance, timestamp, and evidence ID. Represent conflicting evidence explicitly.
- [ ] Show an activity timeline and current action, goal, and remaining budgets.
  Highlight observed additions, consolidation, expiration/eviction, and proof paths
  only when the recorded action or inference provides them; do not invent thoughts.
- [ ] Support pause, single-operation step, rewind/scrub, and replay speed; explain
  the selected step and its before/after state. Separate live/current memory from past
  replay frames: forgotten facts disappear from current memory and are labeled as
  historical when replayed. Display terminal/idle and disconnected states honestly.
- [ ] Verify a synthetic session against underlying memory at each replay tick,
  including expiration, consolidation, conflicts, and restart. Viewer history must
  never enter agent inputs or bypass memory limits; private sessions stay local.
- [ ] Check that enabling recording preserves actions/results under the same seed
  and checkpoint. Capture actual computations without extra stochastic model calls,
  record timing separately from playback speed, and measure recording overhead.
- [ ] Document the launch workflow and demonstrate readable interaction on an
  ordinary laptop, without changing benchmark decisions or requiring paid services.

Acceptance: a user can launch a demonstration, replay it slowly, see which component
operated at each step, understand its inputs and decision, inspect the supporting
facts, and compare memory before/after. This is an observability feature, not evidence
of curiosity or improved learning; a component animation must map to a recorded event.

## M3 — Autonomous investigation
- [x] Persistent tick loop for ASK/ANSWER/ABSTAIN/WAIT, timestamps, teacher/compute budgets.
- [ ] Expand internal computations and learned THINK/RECALL/STORE/REVISE choices.
- [ ] Useful internal computations; idle when no operation has expected utility.
- [x] Restricted human teacher CLI; rejects invalid structured replies.
- [x] Bind restart identity/evidence and validate budgets; detect incomplete ticks before further external calls (automatic recovery remains unfinished).
- [ ] Benchmark hypothesis revision and source reliability.

## M4 — Language track
- [x] Optional frozen local Transformers adapter; lazy loading, no paid APIs.
- [x] Real-model CPU smoke test: frozen weights and cached pages. GPU measurements pending.
- [x] Structured answer rendering, exact entity copying, strict fallback on changed claims.
- [ ] Benchmark language benefit and compare fluency/faithfulness beyond smoke tests.
- [x] Separately invoked local-LLM teacher interface with unverified provenance.
- [ ] Teacher knowledge and linguistic contexts isolated and logged.

## M5 — Token-control experiment
- [x] Cached fixed-distribution pagination, configurable candidate limits.
- [x] Trainable reranker and explicit MORE/END actions; real generation driver.
- [x] Supervised token-loss interface and unit test showing target-selection optimization.
- [ ] Supervised bootstrap; compare claim fidelity, fluency, accuracy, latency, costs.
- [ ] Keep the mechanism only if measured benefits justify complexity.

## M6 — Broader research
- [ ] Learned reasoning versus supplied rules; novel compositions and longer proof chains.
- [ ] Calibration and selective prediction; controlled conflicting sources.
- [x] Multiple optimizer-seed runner and per-world Wilson intervals.
- [ ] Broader vocabulary/distribution shifts and intervals over training seeds.
- [x] Memory graph inspection (Graphviz DOT) and static learning-curve visualization.

Completion requires measured evidence and documented limitations, not just interfaces.
