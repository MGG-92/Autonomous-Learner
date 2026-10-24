# Autonomous development contract

The central objective is an experience-learning cognitive core that discovers and
uses internal models independently of pretrained language knowledge. An optional
LLM translates approved structured observations/messages; it must not supply the
controlled experiment's hypotheses, answers, embeddings, or investigation choices.
Prioritize internal rule/model learning and falsifiable causal tests over further
memory infrastructure, token reranking, or viewer polish. Distinguish learned rules
from supplied primitives/search/inference, and belief updates from parameter training.

The destination is generalizable world learning: objects, properties, actions,
interactions, consequences, and goals. Finite mathematical rules are a diagnostic
starting point, not the objective. Preserve frozen results, but require a concrete
transfer or world-modeling question before further benchmark-specific optimization.
Progress from learned investigation to small structured action worlds and reusable
abstractions; avoid adding perception or language complexity before testing the core.

Functional self-modeling is an explicit long-term objective: learn reliability,
capabilities and effects of cognitive actions from ordinary experience, then test
whether this improves decisions. Follow `docs/SELF_MODELING.md` and C5 in the plan;
preserve the measured development goal baseline and its inconclusive prior comparison.
Next take bounded P1 observability alongside a C5a feedback protocol. No canned
self-descriptions or LLM
self-awareness simulation counts as evidence, and functional benefits imply no
subjective-consciousness result. Preserve the evaluator/agent boundary: any future
internal-observation interface must exclude research audit logs, hidden intervention
labels, held-out answers and forgotten task contents. Test against no-self-model,
simple calibration and matched-resource controls before making self-modeling claims.

The planned Interactive Agent Playground follows `docs/PLAYGROUND.md` and P1 in
`docs/PLAN.md`: first score the action-world goal baseline, then a bounded CLI using
actual cognitive components. Keep manual sessions development-only, observer history
outside agent memory, and step/replay inspection free of repeated decisions. It must
not displace model, strategy, abstraction or self-modeling research with UI work.

Read `PROJECT_STATE.md`, `docs/PLAN.md`, and `docs/ARCHITECTURE.md` before changing code.
The user authorizes autonomous implementation, debugging, tests, sensible architecture
changes, local commits, and pushing this project to its GitHub repository. Do not
request renewed permission for routine work. Never publish credentials or private
teacher conversations. The user explicitly requires this repository to be PUBLIC.
Run `python scripts/publication_check.py` before every push; use a GitHub no-reply
commit email. Only source, tests, documentation, and sanitized synthetic reports
may be published. Never add local paths, private prompts, credentials, databases,
checkpoints, or unreviewed generated artifacts. Do not change
unrelated repositories, account settings, or delete user work.
Keep the README's prominent work-in-progress and agentic-AI disclosure. The user
requires the project to be presented as an experiment, nothing less and nothing more.
Do not rewrite public branch history; the private bootstrap cleanup was one-time.

## Resuming when the user says “continue”

1. Inspect `git status`, the latest commits, and `PROJECT_STATE.md`.
2. Reuse the current checkout. Preserve uncommitted user work.
3. Run the documented smoke tests; take the first unfinished milestone in `docs/PLAN.md`.
   Read the latest entry in `docs/AUDIT_LOG.md` and revisit relevant unresolved findings.
4. Implement, test, and debug a small meaningful increment. Continue while useful work remains.
5. Update `PROJECT_STATE.md` with exact commands, evidence, limitations, and the next task.
6. Use a short-lived feature branch for each meaningful increment, with coherent
   commits. Open a PR against main, review its diff, and require passing CI before
   merging. Routine PR creation and merging are authorized; do not wait for human
   approval unless an unresolved consequential decision requires it. Keep main stable.
   Use merge commits so source hashes in experiment reports remain reachable.
   Record any remote/authentication blocker without delaying independent local work.

## Recurring critique and repair

This is a required development step, not a scheduled background job. At the start
of each continuation, revisit relevant open findings. Before every meaningful
increment's PR, critically review the changed path and at least one adjacent
component: challenge an assumption with a concrete counterexample, inspect the
actual code/evidence, and reproduce suspected defects before claiming a fix.
Check hidden-answer leakage, real forgetting and hard budgets, restart/side-effect
consistency, seed/checkpoint separation, baseline fairness, statistics, and whether
public claims match measured behavior. Choose checks relevant to the increment;
do not add ritual tests or invent findings to fill a checklist.
Also ask which capability could transfer beyond the current task distribution,
what supplied machinery still explains success, and which experiment could refute
the broader interpretation. A higher mathematical benchmark score alone is not
evidence of general world reasoning.

Fix consequential correctness, data-loss, or scientific-integrity defects affecting
the increment before merging, and add meaningful regressions. Explicitly bounded
research limitations may remain open when behavior/claims are honest; record why,
the evidence, and a concrete next investigation. Repeat this review after substantive
repairs. At milestone boundaries and before opening new final experiment worlds,
also review the wider architecture, task distribution, baselines, evaluation protocol,
and outstanding claims. Passing CI is necessary but is not evidence of broad capability.

Append a concise sanitized entry to `docs/AUDIT_LOG.md` with scope/source, findings
(including a no-new-defect result when justified), repairs and checks actually run,
unresolved limits, and the next audit focus. Keep the latest audit/next focus in
`PROJECT_STATE.md`. The initial detailed audit is `docs/CRITIC_AUDIT.md`; preserve
historical negative findings. Never publish private sessions or raw teacher material.

## Scientific rules

- Separate learning from experience (memory) from parameter optimization (policy).
- Oracle internals, gold labels, and held-out answers must never enter policy features.
  Expert trajectories may use oracle internals ONLY in the training module.
- Test worlds use seeds disjoint from training and validation. Never select checkpoints
  or tune rewards on final test results. Record all seeds and hyperparameters.
- Keep a no-memory, random-policy, untrained-policy, and heuristic baseline.
- Report scripted inference, fixed memory admission, and template output honestly.
- Use proper scores for uncertainty experiments; evidence status is not probability calibration.
- Do not label postponed or untested features complete. No fabricated measurements.

## Development

Python/PyTorch, CLI first, CPU-friendly synthetic tests. Use `.venv/Scripts/python.exe`
on Windows or `.venv/bin/python` on Linux. See README for bootstrap commands.
Use `python -m pytest` and `python -m autonomous_learner demo` after relevant changes.
For policy changes run the seeded experiment and save a compact report in `reports/`.
Large checkpoints, caches, models, databases, and raw runs belong in ignored `runs/`.
Keep audit logs inaccessible to the agent; bounded memories must actually forget.
No paid APIs required. Do not schedule background jobs unless requested.
