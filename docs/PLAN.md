# Implementation plan

## M1 — Controlled learning experiment
- [x] Synthetic oracle with randomized names, graph relations, explicit negatives.
- [x] Persistent bounded evidence memory and traceable multi-step inference.
- [x] Agent with structured actions and no access to hidden world state.
- [x] Trainable question/action controller, expert bootstrap, policy-gradient fine-tuning.
- [x] Held-out comparisons, memory ablation, checkpoints, learning curves, CLI demo.
- [ ] Tests, CI, Git history, remote repository, continuation documentation.

## M2 — Learned memory management
- [x] Separate bounded episodic and belief stores (consolidation timing still fixed).
- [x] Task distribution with delayed reuse and controlled distractors; broaden later.
- [x] Learned retention scorer versus FIFO/untrained/heuristic with hard budgets.
- [ ] Explicit consolidation actions and integration with the question policy.
- [ ] Learn retrieval and consolidation timing against budget-matched heuristics.
- [ ] Contradiction handling, source reliability, revision, retention curves.

## M3 — Autonomous investigation
- [ ] Persistent tick loop with THINK/RECALL/ASK/STORE/REVISE/WAIT and resource budgets.
- [ ] Useful internal computations; idle when no operation has expected utility.
- [ ] Human teacher CLI; benchmark multi-turn acquisition and hypothesis revision.

## M4 — Language track
- [ ] Optional frozen local Transformers adapter; lazy loading, no paid APIs.
- [ ] Real-model smoke test on a small configurable model, CPU/GPU cost measurements.
- [ ] Structured answer plans, entity copying, template and frozen-model baselines.
- [ ] Teacher knowledge and linguistic contexts isolated and logged.

## M5 — Token-control experiment
- [ ] Cached fixed-distribution pagination, configurable candidate limits.
- [ ] Trainable reranker and explicit more-candidates/end actions.
- [ ] Supervised bootstrap; compare claim fidelity, fluency, accuracy, latency, costs.
- [ ] Keep the mechanism only if measured benefits justify complexity.

## M6 — Broader research
- [ ] Learned reasoning versus supplied rules; novel compositions and longer proof chains.
- [ ] Calibration and selective prediction; controlled conflicting sources.
- [ ] Multiple training seeds and uncertainty intervals, vocabulary shift, distribution shift.
- [ ] Memory graph inspection and learning-curve visualization.

Completion requires measured evidence and documented limitations, not just interfaces.
