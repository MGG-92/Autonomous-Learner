# Internal rule discovery v1 — bounded learning baseline

Freeze code, configuration, splits, and this protocol before opening final tasks.
Purpose: test observation-driven internal rule learning and informative experiments
without a pretrained language model. This is not a claim of learned inference,
independently invented primitives, or general cognition.

## Public representation and hidden environment

Inputs are integer pairs in 0..4; outputs are in 0..4. A supplied expression language
has leaves x/y/0/1 and binary add_mod, sub_mod, mul_mod (modulo five), min, and max.
Generate all expressions of depth at most two, with one canonical representative
per full-domain output table: 1210 distinct functions. This semantic quotient avoids
counting syntactic aliases as extra prior votes or as independent held-out tasks.
The canonical representative is not necessarily the environment's original syntax.

The learner starts with a uniform prior over these hypothetical functions. It has
no oracle rule ID, hidden program, partition label, or held-out targets as an input.
Each queried example removes inconsistent hypotheses. This is learning through
evidence-dependent belief updating, with no parameter optimization. Primitives,
enumeration, filtering, prediction, and information-gain selection are supplied.
The independent core imports neither Torch nor Transformers nor language adapters.

Inference considers all public hypotheses, including potential final rules. This
is a known hypothesis class, not a claim that the class itself was learned. Its
precomputed hypothetical predictions must not be confused with observed oracle data.

## Disjoint rules and matched evaluation

Hash each full 25-value function signature using SHA-256. The first eight hex digits
modulo 100 assign development (<70), validation (70..84), or final (85..99), except
the nine named reference functions (x, y, constants 0/1, and each binary primitive
on x/y), which are development-only regardless of hash. This excludes the actual
max/projection software examples from held-out rules. No equivalent function
crosses partitions. No models
or policies are pretrained on development tasks in this first baseline; future
learned search policies may use development tasks under a new protocol.
The resulting partition counts are 846 development, 165 validation, and 199 final.

Sample 100 distinct rules without replacement in the chosen partition using seed
17. For rule index i, use an independent layout RNG seed 3200000+i for validation,
3300000+i for final, to reserve five input pairs. Layout seeds depend on index,
never hidden function values/IDs. All policies/seeds in that world share the same
20 legal query inputs and five reserved prediction inputs. Four queries are allowed;
stop early only when the full finite function is uniquely identified or no valid
model/query remains. Report actual queries. Do not evaluate each policy on a
different complement of its own queried inputs.

Query seeds 7/42/97 affect information-gain ties and random exploration, not training.
Report 100 unique semantic rules, not 300 independent worlds or optimizer repeats.
Predictions are committed before the evaluator reads reserved targets.

## Controls and metrics

- Information gain: supplied heuristic maximizing predictive entropy. For uniform,
  deterministic hypotheses this equals expected information gain about function identity.
- Random: same belief updates, uniformly chosen unqueried legal inputs.
- Fixed/untrained policy: same updates, lexicographically ordered queries. There is
  no trained neural policy here; do not pretend this is an untrained neural ablation.
- Model reset: after active investigation, restore the prior and erase raw responses,
  retaining only the asked-input set. Re-evaluate the same targets without new queries.
- No memory: discard the same active replies and predict from a fresh prior. Its
  predictions equal reset by construction; these are not two independent findings.

Report accuracy (argmax ties favor smaller outputs), multiclass Brier score (sum of
squared errors over five outputs), natural-log loss, actual query count, and unique
full-function identification. Prediction success alone does not imply identification.
Report mean paired Brier reductions for active versus random/fixed/reset. Average
query seeds within semantic rule, then bootstrap rules with 2000 resamples, seed 197,
percentiles at sorted indices 50/1950. Intervals are exploratory and unadjusted;
they do not cover optimizer uncertainty, new primitive families, or other domains.
There is no new arbitrary pass/fail threshold or automatic claim of general reasoning.

An empty version space is explicitly MODEL_CLASS_FAILURE, not high-confidence
fallback. Main evaluation deliberately uses in-class reliable deterministic rules;
misspecification is a software counterexample, not a calibrated noisy-world benchmark.

## Limits and next evidence

This tests bounded program induction and active learning. The 5x5 domain, depth-two
language, human primitives, uniform prior, and supplied algorithms are strong biases.
The hash split holds out functions, not families/primitive types. No extrapolation
to arbitrary integers, invented reasoning strategy, source reliability, or autonomous
goal formation is demonstrated. Posterior values are relative to this model class;
proper scores do not by themselves establish calibration. Informative observations
can leave ambiguity: keep it visible rather than selecting a rule without evidence.

Query ceilings are matched, but computation is not: information gain scans candidate
predictions while random/fixed selection is cheap. Report observation efficiency,
not a compute-efficiency advantage. The first report gives total evaluation time,
not a per-policy performance benchmark.

Next compare learned search/query proposals with this baseline under matched
observation/computation costs, then induce reusable abstractions. Reserve a new final
partition/protocol before evaluating revised algorithms. Keep the LLM absent until
the core's model dependence is established; later test rendering/parsing separately.

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery-smoke.json --output runs/rules-smoke
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery.json --output runs/rules-validation
.\.venv\Scripts\python.exe -m autonomous_learner rule-discovery --config configs/rule-discovery.json --output runs/rules-final --phase final
```

Final is reserved until source/config freeze. Do not tune on opened final functions.
