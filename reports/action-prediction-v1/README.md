# Action-world prediction: first development comparison

Source/config/protocol freeze: `b94d48acb3880e88e244f093cf196ef073cfef3f`, clean tree.
This is development-only evidence, not a final test, learned reasoning algorithm,
goal-achievement result, or transfer experiment. See the
[protocol](../../docs/experiments/structured-action-world.md) and
[compact synthetic report](development.json). Previous reports are unchanged.

## Experiment

Sample 80 distinct canonical development mechanisms uniformly, independently shuffle
object/action slots and names, then allow 12 sequential real actions from the same
start per policy. No arbitrary state preparation or exploration reset is available.
Compare supplied one-step information gain, random actions and fixed cyclic actions;
all update the same transition model. The reset/no-memory/untrained control erases
the information-gain run's model and raw replies before prediction, reusing its trace.
No policy parameters are optimized. Filtering, effect templates, priors and sequence
inference are supplied; learning means conditioning hypotheses on experienced effects.

Each world has a shared 24-case one-step panel and 16 sampled two-step cases. All
policies and RNG seeds 7/42/97 commit every prediction before any panel target is
read. Panels include hypothetical states that can be unreachable from exploration.
Common-unseen cases exclude experience from every policy AND seed in a world.
These are conditional diagnostic cases, not an independent test distribution.

## Results

Means across the three policy RNG seeds, with equal world weights. Seeds share
the same 80 worlds and probe panels; repeated predictions are not independent worlds.
Lower joint-state Brier is better; full individual-seed results are in the JSON.

| Investigation / model | One-step accuracy | One-step Brier | Two-step accuracy | Two-step Brier |
|---|---:|---:|---:|---:|
| Information gain + updated model | 75.31% | 0.3407 | 71.80% | 0.3837 |
| Random + updated model | 73.63% | 0.3672 | 69.97% | 0.4131 |
| Fixed cycle + updated model | 73.33% | 0.3680 | 69.84% | 0.4026 |
| Model reset / no memory / untrained prior | 61.98% | 0.5682 | 56.41% | 0.6074 |

On the common-unseen panel:

| Investigation / model | One-step accuracy | One-step Brier | Two-step accuracy | Two-step Brier |
|---|---:|---:|---:|---:|
| Information gain + updated model | 68.08% | 0.4326 | 64.55% | 0.4767 |
| Random + updated model | 67.91% | 0.4499 | 63.79% | 0.4990 |
| Fixed cycle + updated model | 67.71% | 0.4497 | 63.26% | 0.4872 |
| Model reset / no memory / untrained prior | 61.35% | 0.5741 | 55.08% | 0.6132 |

Common-unseen coverage averages 17.325/24 one-step cases and 12.075/16 two-step
cases per world. None of these panels was empty in this run. Of the 966 unseen
two-action cases, 47 consist of two single transitions seen somewhere in the
union of compared experience; unseen sequences do not necessarily have unfamiliar
components. Coverage and all denominators are preserved per world in the report.

Paired Brier reductions average seeds within each world before 2000 world-bootstrap
resamples. Common-unseen information gain versus random: one-step +0.0173,
exploratory 95% interval [0.0087, 0.0274]; two-step +0.0223 [0.0110, 0.0355].
Versus reset: +0.1415 [0.1093, 0.1723] and +0.1365 [0.1018, 0.1717].
The two-step unseen comparison against fixed cycling is +0.0105
[-0.0055, 0.0289], crossing zero. Do not claim a universal investigation advantage.
Intervals are unadjusted and cover this development population/conditional panel,
not new families, optimization seeds or real-world performance.

## Critique and limits

The updated internal model improves predictions over deleting experience. The supplied
information-gain rule offers a modest observation-efficiency benefit over random
here. Neither result shows that the agent learned an investigation algorithm,
invented representations, or transferred acquired knowledge between worlds.
The independent labeled-effect prior differs from the uniform canonical-world
population; proper scores alone do not establish probability calibration.

No-op outcomes dominate the exploration: information gain averages 9.0625 no-ops
out of 12 actions. An offline post-hoc stasis check found that the prior's argmax
always predicts the unchanged state. Its accuracy therefore equals a simple
persistence forecast: 1190/1920 one-step targets and 722/1280 two-step targets stay
unchanged. This is an interpretation check on saved targets, not a newly tuned
policy or additional experiment. Accuracy alone would overstate understanding;
the updated-model comparisons and proper uncertainty scores matter.

The three investigation controls each use 12 real actions; compute is measured,
not equalized. There are 720 actual investigation runs (8640 real actions) and
240 additional reset prediction records reusing their source traces. Mean selection
effect evaluations are 761.5 for information gain and zero for random/fixed. Total
exploration work is 1043.1 versus 293.6/291.3; prediction work is 6703.0 versus
7212.5/6968.4. The reset predictor uses 45634.5 evaluations and shares the original
exploration cost/trace. Counters exclude population construction and evaluator gold
simulations. Local run time was 17.96 seconds including 0.13 seconds construction;
timings are descriptive and no compute-speedup claim follows.

Next: supplied open-loop goal planning using the acquired model, with independent
goal sampling and prior/random-plan controls. Preserve this result and freeze the
next scope before measurement. Learned search, new abstractions, noise revision,
mechanism/family transfer and final evaluation remain open.

## Verification

Command: `python -m autonomous_learner action-prediction --config configs/action-prediction.json --output runs/action-prediction-v1`.
121 tests and the scoring smoke passed before the source freeze. All 3840 score
rows and 960 cost records were checked offline against saved committed predictions,
joint targets, traces and union-based unseen sets. Every aggregate, paired difference
and bootstrap interval was reproduced without rerunning worlds. Raw records remain
ignored under `runs/`; only the reviewed compact synthetic report is published.
