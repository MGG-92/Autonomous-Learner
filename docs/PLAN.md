# Implementation plan

## Active direction — Neural Volume Lab (explicit user redirection)

First settle the visual embedding system in a native, launchable desktop app, then
investigate the spatial neural network. Work in `neural-volume-lab/`; preserve all
older code/results. The pretrained embedding source is explicit and is not evidence
of independently acquired language knowledge. See that folder's README for launch.

- [x] E1: native dark coordinate volume, actual frozen token embeddings, consecutive
  XYZ polylines, shared scale, token-by-token playback/selection and camera controls.
- [x] E1: message/optional expected-response reference, raw-vector inspector and
  explicit labels distinguishing reference visualization from training/prediction.
- [x] E1: offline model export, checksummed read-only assets, native widget tests and
  Windows packaging/launch entry point; packaged executable verified end to end.
- [ ] E2: settle NN stimulation and sequence encoding; define trainable silent
  initialization, directed recurrence, update timing, stability and text readout.
- [ ] E2: implement/visualize actual NN states, then test fitting and interference
  on user-taught associations. No teacher, spontaneous activity or learning claims yet.

## Preserved cognitive-core roadmap

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

Long-term objective also includes learning how the agent itself works: empirical
knowledge of reliability, capabilities, limits and effects of cognitive operations.
This is functional self-modeling, not a consciousness claim. The development goal
baseline is measured; next take bounded P1 and specify a small C5a reliability pilot; C3 abstractions and C5 can inform one
another. C4 language rendering is not a prerequisite for C5. Details and controls:
`docs/SELF_MODELING.md`. Do not add a narrative self-awareness layer.

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
- [x] Test unseen interventions and action sequences, then goal achievement using
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
`configs/action-world-development.json` (retained original proposal), with
the measured prediction-only subset in `configs/action-prediction.json`. The simulator
has three binary-property objects and three unknown guarded effects; software tests
cover sequential interactions, action budgets, public-record boundaries and semantic
identity under all object/action renamings. The sampler and transition-belief core
are now implemented, with a scripted `action-model-demo`. The core learns by belief
updating within supplied templates, without parameter training. The first 80-world
development prediction comparison is recorded in `reports/action-prediction-v1/README.md`.
No validation/final or transfer result is established. The supplied goal runner
is now scored under `configs/action-goals.json`; see `docs/experiments/action-goals.md`.

Results: `reports/action-goals-v1/README.md`. Across 80 development goals (only
10 reachable), IG success is 10.42% versus prior 6.67%, random investigation 10.83%
and fixed 10.42%. The paired prior-gain interval crosses zero; no reliable prior
advantage or investigation superiority is established. Planning uses supplied search
over acquired dynamics, not a trained planner. Preserve this sparse-panel limitation;
a broader goal study needs a separate protocol before measurement. Next take bounded
P1 observability alongside specifying C5a's legitimate feedback schema.

### P1 — Interactive Agent Playground (planned)

After the first scored action-world goal baseline, implement one bounded CLI
increment using the real simulator, acquired transition beliefs, investigation
controls and planner. Intended command: `python -m autonomous_learner playground`
(not implemented). This is a manual development sandbox, not a scientific benchmark.
Do not delay C5a or C3 for interface polish, arbitrary language or graphics.

- [ ] Expose structured goals, prediction/inspection queries and step/run execution
      through the same cognitive components, with evidence, ambiguity and budgets.
- [ ] Emit versioned actual-operation events; inspection/replay must not rerun
      decisions, consume observations or expose researcher history to the agent.
- [ ] Label experience-updated beliefs, supplied algorithms, trained components
      and optional LLM operations separately; do not simulate missing capabilities.
- [ ] Enforce development-only mechanism isolation and local session storage;
      manual sessions cannot feed reserved evaluation or automatic training.
- [ ] Verify recording and step/run parity, document a real-component walkthrough,
      then resume research. Add memory/abstraction/self-model views as implemented.

Scope, extension boundaries and future acceptance checks: `docs/PLAYGROUND.md`.
Rule editing and graphical interaction are later increments. Existing stationary
models may fail after changed rules; do not imply adaptive revision already exists.
The historical M2.5 graphical viewer can share these events; its delayed-memory
integration dependencies do not block this action-world CLI.

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

### C5 — Functional self-modeling and metacognition (research, not implemented)

After the first scored goal baseline, start C5a without waiting for language or
general abstractions. Later stages depend on measured benefit, not persuasive output.

- [ ] C5a: define a bounded, legitimate internal-observation/ordinary-feedback
      interface; train an empirical error/success forecaster. Compare raw task confidence,
      base-rate/scalar calibration, fixed thresholds and task-only predictors. Separate
      reliability estimation from task success, calibration and metacognitive control.
- [ ] C5b: predict agent-specific memory/strategy/resource effects under independently
      varied tasks and constraints. Control task difficulty and strategy-selection bias;
      test shuffled/stale/other-agent histories and novel capability combinations.
- [ ] C5c: use learned forecasts to investigate, retrieve, reconsider, switch or stop.
      Compare no-self-model, monitor-only, fixed heuristic and capacity/information-matched
      history-based controllers. Require net benefit after all resource costs; ablate
      the self-model without deleting the task model.
- [ ] C5d: connect useful metacognitive control to C2 strategy learning and C3 acquired
      abstractions/update rules; compare from-scratch/exact initialization and hold out
      task families and internal constraints. No automatic code self-modification needed.

All stages need their own development/final protocol and explicit feedback/memory
boundaries. Existing uncertainty counts, research logs and memory-controller training
do not demonstrate a learned self-model. Preserve negative outcomes and reject
claims explained solely by extra feedback, memory, compute or intervention labels.

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

P1 above is the first, smaller interactive playground. This retained graphical
viewer is a later optional extension using shared actual-operation events. The
historical dependencies below apply to the graph-memory viewer, not to P1.

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
