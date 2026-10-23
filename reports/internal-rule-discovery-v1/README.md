# Internal rule discovery — first bounded result

An agentic-AI experiment in learning an internal model from experience, with no LLM
involved. Frozen source: `aedf284f0bf84a4accf3ccaab827f9a6f8af728d`, clean tree.
[Code/config/protocol](../../docs/experiments/internal-rule-discovery.md) and semantic
splits were frozen before final evaluation. No parameters or search policy were trained.

## What actually learns

The core starts with 1210 generated executable functions in a supplied depth-two
language over integer pairs 0..4. Queried input/output examples eliminate inconsistent
models. A supplied information-gain heuristic chooses queries. This is bounded
rule induction through belief updating; primitives, enumeration, inference, and
investigation strategy are programmed. It is not invented logic or general cognition.

The core can operate without importing Torch, Transformers, or language adapters.
Predictions are committed before the evaluator reads reserved outputs. Structured
messages describe model predictions; no LLM is asked to solve or explain the task.

## Frozen final evaluation

100 distinct semantic functions, five reserved inputs each, and query seeds 7/42/97
shared across policies. These are 100 unique rules and 500 related reserved inputs,
repeated across three exploration seeds, not 300 independent worlds or training runs.
Every policy has the same reserved targets and a four-query ceiling.

| Policy/control | Accuracy | Brier score (lower better) | Log loss | Mean queries | Full-function identification |
|---|---:|---:|---:|---:|---:|
| Supplied information-gain investigation | 76.53% | 0.2460 | 0.3639 | 4.000 | 25.00% |
| Random queries, same model updating | 66.60% | 0.4007 | 0.6749 | 3.990 | 10.33% |
| Fixed/untrained query order, same updating | 54.40% | 0.5360 | 1.0214 | 3.980 | 6.00% |
| Reset model and erase responses after investigation | 28.20% | 0.7804 | 1.5633 | 4.000 | 0.00% |
| No memory, discard the same active replies | 28.20% | 0.7804 | 1.5633 | 4.000 | 0.00% |

Identification rates average rule/query-seed runs. Correct predictions often coexist
with multiple plausible rules. No-memory and reset predictions are equal by
construction; they are not independent confirmations. No trainable controller
exists here: fixed query order is an untrained control, not an untrained neural model.

Mean paired Brier reductions for active investigation are +0.1547 versus random
(95% rule-bootstrap [0.1086, 0.2007]), +0.2901 versus fixed [0.2279, 0.3514], and
+0.5344 versus reset [0.4960, 0.5706]. Query seeds are averaged within each rule
before 2000 rule resamples, bootstrap seed 197. Intervals are exploratory and
unadjusted, and do not cover training-seed or new-domain uncertainty.

This supports two scoped conclusions: observations improve predictions through the
internal model, and the supplied information-gain strategy gives better predictions
under the same query ceiling than these controls. It does not show that the strategy itself was learned.
Computation is not matched: information-gain selection does more work than random/fixed.

## An actual internal-model trace

The first recorded final example (layout 3300000, query seed 7):

| Query | Observed output | Remaining semantic hypotheses |
|---|---:|---:|
| (4,4) | 1 | 247 |
| (3,1) | 2 | 46 |
| (2,3) | 0 | 5 |
| (4,3) | 1 | 1 |

Within this finite language/domain, the remaining function has representative
`min(add_mod(x,y),sub_mod(0,x))`: take the smaller of `(x+y) mod 5` and `(-x) mod 5`.
It predicts reserved inputs (0,1), (1,3), (0,0), (3,0), (4,1) as 0,4,0,2,0 before
evaluation; those match the oracle. This is one recorded example, not a success
rate or a guarantee of unique discovery. The canonical expression need not match
the environment's original syntax, and equivalence is guaranteed only on the finite domain.

## Critique, split correction, and limits

Before final evaluation, the recurring audit found that named primitive/max/projection
software examples could hash into the final partition. All nine reference functions
were moved to development-only; four changed partition. The resulting semantic split
is 846/165/199, with equivalent expressions never crossing splits. Validation results
were unchanged; no final outcome informed this correction. Another test falsified an
overconfidence assumption: four observations of max leave six compatible functions,
so the core retains uncertainty rather than claiming to have uniquely identified it.

Full development validation accuracy was 72.6% active, 66.8% random, 54.0% fixed,
and 25.6% reset/no-memory. `validation.json` is explicitly a dirty-tree development
run. `final.json` records the clean frozen source, all 1500 synthetic per-rule rows,
configuration, exact summaries, comparisons, and the example above. Local suite:
77 tests passed; the legacy graph demo also passed. Final evaluation took 1.27 seconds
after hypothesis construction on the development CPU; this is not a portable benchmark.

The finite grammar is a strong human prior, the environment is reliable/deterministic,
and final rules remain in that grammar. No new primitives/families, open-ended
concept learning, noisy belief revision, learned reasoning strategy, autonomous goal
formation, or language benefit is established. Posterior probabilities are relative
to the assumed class; proper scores do not establish real-world calibration.

The selected final functions/layouts are now OPENED and must not be used for tuning.
Next: train search/query guidance on development tasks and compare against this
explicit baseline under matched observation/compute costs with a fresh final protocol;
then investigate reusable abstractions. Preserve this result and its limitations.
