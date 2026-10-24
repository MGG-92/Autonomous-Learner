# Action-world open-loop goals: development protocol v1

This adds goal scoring to the existing structured action-world bridge. Preserve
the frozen prediction protocol and results. No validation/final worlds, policy
optimization, calibration claim, family transfer or self-model is involved.

Question: does acquired transition evidence improve short goal-directed plans over
the same supplied planner with erased evidence? Investigation controls separate
information gain, random and fixed cyclic exploration; a random goal plan controls
the value of planning. Effect templates, filtering and search remain supplied.

## Frozen settings and task sampling

Use `configs/action-goals.json`: the same 80 distinct development mechanisms selected
with seed 611, exploration/name/permutation ranges as the prediction comparison,
12 exploration actions, goal budget three, and controller RNG seeds 7/42/97.
These are not new held-out mechanisms. Each policy starts fresh in every world.
Goal-start states use seeds 5300000 + world index; goals use 5400000 + index and
are uniform over the seven other full states. Selection receives no mechanism,
reachability, gold path or exploration outcome. One shared goal per world is a
limited diagnostic. No resampling of unreachable goals or absorbing worlds.

Investigation uses the existing controls and `(seed << 32) ^ exploration_layout_seed`.
Goal tie RNG uses `(seed << 32) ^ (5500000 + index)` independently of investigation.
Every policy/seed shares mechanism, naming/permutation, exploration start, goal start
and goal. Fresh goal episodes are evaluator-controlled boundaries, not free agent
resets during investigation. No language model or learned policy weights are loaded.

## Commit, execute, then analyze

After each investigation, freeze acquired evidence. Exhaustively rank the 40 sequences
of length zero through three by model-relative goal probability, shorter tied sequence,
then public seeded choice. Persist all plans for every policy AND seed before true
reachability analysis or goal execution. `execute` has no model argument and cannot
update beliefs or replan; it only executes the committed sequence and validates replies.

Controls: information-gain/random/fixed exploration each feed their updated model to
the same planner. Model reset reuses the information-gain exploration trajectory and
cost, clears posterior and raw evidence, then plans from the prior. This is also the
no-memory/untrained control: there is no parameter training. Random-plan reuses that
same exploration cost but ignores its evidence and independently samples each of
three action slots uniformly (27 equally likely length-three sequences). It has no
predicted success estimate. Do not treat either reused trace as extra exploration.

After commitment the evaluator computes bounded true reachability (including paths
shorter than the ceiling) and executes each plan in a separate identical goal episode.
Report exact final-state success on ALL sampled goals first; reachable/unreachable
subgroups include their actual denominators and unavailable rates for empty groups.
Do not expose reachability to the planner or drop failures from primary performance.
Predicted success is model-relative, not calibrated self-reliability.

## Costs, uncertainty and artifacts

Report per-seed world-macro success and actual goal actions. Report exploration
selection/filtering work, plan effect evaluations, evaluated sequences, no-op counts
and stage wall times separately. Goal ceilings match but actual lengths may differ;
compute is measured, not matched. Population construction and evaluator reachability
work are separate from learner costs. Reset/random-plan inherit IG exploration costs.

Pair success differences within world, average seeds within world, bootstrap those
80 world differences 2000 times using seed 197 and sorted indices 50/1950. Report
per-world differences and individual seeds. Intervals are exploratory/unadjusted,
conditional on development tasks; repeated seeds are not independent worlds.
No success threshold or later model selection is defined from this measurement.

Freeze source/config/protocol in Git before the measured run. Record commit, dirty
status, all seeds, Python and elapsed time. Keep full commitments, ordinary transition
traces, execution outcomes, scores and costs under ignored `runs/`; publish only a
reviewed compact synthetic report. Existing reports and protocols stay unchanged.
Tests must verify cross-seed commitment order, reset erasure, no goal model updates,
no-op budgets, unreachable goals, reproducibility and independent goal sampling.

Run: `python -m autonomous_learner action-goals --config configs/action-goals.json --output runs/action-goals-v1`.
Smoke config uses two development worlds, one seed, three exploration actions and
a one-action goal ceiling. It checks software, not the 80-world result.
