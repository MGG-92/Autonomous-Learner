# Critical audit — 2026-10-23

This audit examines the source, restart contracts, and published synthetic evidence.
It distinguishes repaired software defects from unproven research capabilities.
The project remains an agentic-AI experiment.

## Reproduced defects and repairs

| Finding | Consequence before repair | Repair and verification |
|---|---|---|
| Memory construction trimmed to the requested capacity | Opening an existing database with a different budget could delete evidence before session compatibility checks | Persist schema/capacity metadata; refuse incompatible opens without trimming. A regression checks byte-for-byte preservation. Legacy databases bind their first explicit capacity only if no evidence needs deletion. |
| Session identity omitted both controller weights and retention settings | A resumed run could silently change its decision process | Schema-v2 sessions bind weight digests, feature names, task, budgets, and both memory capacities. Weight-matched new instances resume; changed question or retention weights are rejected. |
| Session counters were trusted and evidence files were not bound to them | Invalid budgets, a missing database, or interrupted writes could produce a misleading continuation | Validate counters/history/timestamps, bind the current evidence digest, and preflight expected files in the CLI before opening databases. Missing/replaced memory is rejected. |
| Teacher calls preceded persisted intent | A process crash after a call or database commit could cause the action to run again | Save a pending action and reserve its teacher call first; clear intent only in the completed session commit. Injected commit failure and teacher errors stop both live and restarted loops before another call. |
| Multi-fact storage had no rollback boundary | A failed insert could leave a partially admitted observation accessible | Store the observation and capacity eviction in one SQLite transaction. A trigger-induced failure verifies full rollback. |
| Fact polarity was only annotated as boolean | A string such as `"false"` could become truthy evidence | Validate facts, queries, observations, provenance, timestamps, and teacher-query correspondence at their entry points. Invalid teacher facts do not enter memory or asked-history. |
| Delayed snapshot validation inspected deduplicated visible evidence | An invalid belief record could hide behind a newer episodic copy | Check every underlying record, integer budgets/ticks, episode order, and consolidation consistency before exposing loaded state. |
| Paired statistics overwrote repeated rows and allowed incomplete seed coverage | An invalid dataset could silently change a comparison or reweight optimizer seeds | Reject duplicate measurements, missing pairs, inconsistent seed sets, empty data, and nonfinite utilities. Recomputing the published reference comparison gives exactly the existing result. |
| Configuration comparisons accepted NaN and fractional/boolean counts | Invalid optimization settings could pass preflight | Require finite optimization settings and nonnegative integer seeds/counts/budgets before training. |

Seventeen initial reproductions failed on the pre-repair source. The complete suite
now has 64 passing tests, including crash injection, rollback, unchanged-weight
resume, missing-database CLI preflight, and preservation of legacy files. Existing
question-learning and delayed-memory tests still pass. CI also exercises a two-call
persistent CLI run, rather than testing only its Python API.

Local smoke evidence: a fresh-database demo passed; a heuristic session resumed
after two ticks and answered after five total; actual saved question/retention
controllers did the same through the CLI. The seeded original question smoke reached
30/30, with random/untrained/no-memory 0/30. The small delayed validation smoke
completed with its expected negative gate (learned accuracy 0/6); completing a smoke
does not mean its learning gate passed. No final worlds were reopened or used to tune.

## Compatibility and residual engineering limits

Sessions now use schema version 2. Old sessions lack enough identity information
to certify a continuation: preserve them and use their original source version,
or start a fresh session/database pair. There is no implicit migration, deletion,
or budget reset. Existing standalone SQLite databases gain capacity metadata on
their first compatible open; an over-capacity database is refused intact.

An interrupted tick is detected and refused; there is no automatic recovery yet.
Preserve its JSON, SQLite files, and audit output for investigation. Never clear
the pending marker merely to replay an uncertain teacher call. SQLite evidence and
session JSON remain separate commits; this change does not make them one atomic
transaction. Filesystem power-loss durability and simultaneous writers are not
guaranteed. Use one writer per session. Audit output can lag a committed tick and
is never a recovery input to the policy. Digests detect accidental mismatch; they
are not signatures or a defense against an actor who edits both data and digests.

Persistent execution supports the deterministic question Controller and Heuristic.
Stochastic policy RNG state is not persisted, so unsupported controllers are refused
instead of claiming reproducible restart. Teacher implementation type is bound;
this does not authenticate a human source or prove that external teacher notes or
local model files have remained identical. Memory/controller/schema changes need
an explicit compatibility decision in future versions.

## Research criticism that these repairs do not resolve

1. **The task is narrow.** M1's perfect accuracy uses an easy fixed graph family,
   engineered features, and supplied inference. It is not evidence of learned logic,
   curiosity, autonomous scientific reasoning, or general language understanding.
2. **Timing generalization fails.** Frozen delayed policies answer 0% with two-tick
   observation gaps. More write budget can favor the graph heuristic. Input guards
   and restart fixes do not repair those learned behaviors. Train on a separate
   mixed timing/resource curriculum, select on separate validation worlds, then
   reserve new final worlds before evaluating any claimed improvement. Both opened
   delayed final ranges remain barred from tuning.
3. **The gate is scoped.** Beating FIFO and initialization does not mean beating a
   strong graph method robustly. The fresh reference comparison includes zero;
   intervals are exploratory across worlds, not uncertainty across optimizer seeds.
4. **Memory management is incomplete.** Admission count is fixed, retrieval is
   supplied, and the delayed controller is not integrated into investigation or
   jointly trained with questioning. Beliefs have no time decay. Longer quiet tails
   alter timing features/schedules; they do not demonstrate long-term retention.
5. **Trust and uncertainty remain unmeasured.** An unverified provenance label is
   not learned source reliability, and evidence status/action scores are not
   calibrated probabilities. Keep contradictory/unreliable-source tasks and proper
   scoring rules as explicit future experiments.

The repaired defects have executable regressions. The remaining items require new
implementation and evidence and remain unchecked in the plan. Preserve the negative
reports rather than replacing them with claims that the system is now broadly robust.
