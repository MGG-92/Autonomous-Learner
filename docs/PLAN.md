# Implementation plan

## M1 — Controlled learning experiment
- [x] Synthetic oracle with randomized names, graph relations, explicit negatives.
- [x] Persistent bounded evidence memory and traceable multi-step inference.
- [x] Agent with structured actions and no access to hidden world state.
- [x] Trainable question/action controller, expert bootstrap, policy-gradient fine-tuning.
- [x] Held-out comparisons, memory ablation, checkpoints, learning curves, CLI demo.
- [x] Tests, CI, Git history, public remote repository, continuation documentation.

## M2 — Learned memory management
- [x] Separate bounded episodic and belief stores (consolidation timing still fixed).
- [x] Task distribution with delayed reuse and controlled distractors; broaden later.
- [x] Learned retention scorer versus FIFO/untrained/heuristic with hard budgets.
- [x] Integration of frozen learned retention and question controllers in tick loop.
- [ ] Explicit learned consolidation actions and joint training of policies.
- [ ] Learn retrieval and consolidation timing against budget-matched heuristics.
- [ ] Contradiction handling, source reliability, revision, retention curves.

## M3 — Autonomous investigation
- [x] Persistent tick loop for ASK/ANSWER/ABSTAIN/WAIT, timestamps, teacher/compute budgets.
- [ ] Expand internal computations and learned THINK/RECALL/STORE/REVISE choices.
- [ ] Useful internal computations; idle when no operation has expected utility.
- [x] Restricted human teacher CLI; rejects invalid structured replies.
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
