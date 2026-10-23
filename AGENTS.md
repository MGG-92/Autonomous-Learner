# Autonomous development contract

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

## Resuming when the user says “continue”

1. Inspect `git status`, the latest commits, and `PROJECT_STATE.md`.
2. Reuse the current checkout. Preserve uncommitted user work.
3. Run the documented smoke tests; take the first unfinished milestone in `docs/PLAN.md`.
4. Implement, test, and debug a small meaningful increment. Continue while useful work remains.
5. Update `PROJECT_STATE.md` with exact commands, evidence, limitations, and the next task.
6. Commit coherent increments with descriptive messages; push if authenticated.
   Record any remote/authentication blocker without delaying independent local work.

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
