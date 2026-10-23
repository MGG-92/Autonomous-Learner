# Learned investigation: first development comparison

This is a bounded negative/mixed result, not a final evaluation or evidence of
general reasoning. Source/config/protocol freeze:
`bcd7f5106425b18db217c049e0cb8a09435b20f0`, clean tree. No parameter or checkpoint
selection used validation. See [protocol](../../docs/experiments/learned-investigation.md)
and [complete compact report](validation.json).

## What was learned and supplied

A small neural query scorer was optimized from the prediction improvement caused
by its training observations. No information-gain labels, query demonstrations,
language model, pretrained representations, or validation targets entered training.
Hidden training targets supply a Brier-improvement reward, not policy features.
The finite grammar, exact model filtering, probability features, reward, and optimizer
are supplied. This changes query-policy parameters; it does not learn inference,
invent primitives, discover abstractions, or demonstrate action-world reasoning.

Train 800 episodes on development functions, then evaluate the last weights on
80 distinct shared validation functions with five reserved inputs each, at most
four queries. Optimizer seeds 7/42/97 share world selection/layouts. The 400 reserved
inputs are clustered within 80 functions; seed repeats are not independent worlds.
All settings, seeds, training curves, per-world paired Brier differences, and example
query traces are in the compact report. Final functions were not evaluated.

## Results

Accuracy is percent; lower Brier is better. Shared controls repeat exactly across
seeds except descriptive timing.

| Policy | Accuracy, seed 7 | Accuracy, seed 42 | Accuracy, seed 97 | Brier, seed 7 | Brier, seed 42 | Brier, seed 97 |
|---|---:|---:|---:|---:|---:|---:|
| Trained | 74.75 | 76.75 | 73.50 | 0.2895 | 0.2787 | 0.3196 |
| Exact initial network | 75.50 | 74.25 | 77.75 | 0.2866 | 0.3404 | 0.2656 |
| Information gain | 76.50 | 76.50 | 76.50 | 0.2658 | 0.2658 | 0.2658 |
| Random | 70.25 | 70.25 | 70.25 | 0.3859 | 0.3859 | 0.3859 |
| Fixed | 54.50 | 54.50 | 54.50 | 0.5410 | 0.5410 | 0.5410 |
| Trained, prior query features | 72.00 | 72.00 | 67.75 | 0.3411 | 0.3411 | 0.4009 |
| Trained, prediction model reset / no memory | 26.25 | 26.25 | 26.25 | 0.7848 | 0.7848 | 0.7848 |

Trained mean accuracy is 75.0%, versus initialization 75.83%. Mean Brier is
0.2959 versus 0.2975, hiding opposing seed effects: training worsens Brier in
seeds 7 and 97 and improves seed 42. A reliable benefit over initialization is
not established. All trained seeds trail information gain on Brier, even though
seed 42 has slightly higher accuracy. Beating random is insufficient to attribute
success to optimization: random initial networks already rank these supplied
uncertainty features usefully. No significance or calibration claim is made.

Using updated model features improves Brier against prior-only query features in
all three seeds (paired mean reductions 0.0516/0.0624/0.0813). Resetting the prediction
model also hurts. These are scoped model-dependence checks, not proof of general
reasoning. Prior-feature selection still knows queried actions, remaining budget,
and the common unique-identification stopping rule; it removes adaptive posterior
features specifically. No-memory and post-query reset are equivalent predictions,
not independent controls. The reset preserves the trained policy's query trajectory.

## Cost and interpretation

Training took 5.67/5.86/5.87 seconds locally, with 30.43/30.52/30.46 million
hypothetical output reads. Trained query selection averages about 30013 reads per
world, versus information gain 29988; both scan the full remaining query set.
Random/fixed selection scans no hypotheses, though filtering and prediction do.
Mean selection wall times were about 2.93 ms trained and 2.82 ms information gain.
These timings are descriptive only. Counts exclude neural/optimizer arithmetic and
shared catalog construction; compute was measured, not equalized. There is no
speedup or training-amortization benefit demonstrated.

The marginal-probability representation cannot distinguish actions with identical
predictive masses but different relationships to future actions. It is a supplied
restriction on possible strategies. Optimization alone did not resolve the larger
question of learning transferable investigation. Preserve this first comparison;
do not choose a favorable seed or reopen final arithmetic tasks to tune it.

Next: specify the minimal structured action-world development protocol, where
transition predictions, interventions, and short goal-directed action sequences can
test a broader capability. Keep the C2 compute-matched and family-transfer questions
open. Do not label search-strategy learning complete from these results.

## Verification

Command: `python -m autonomous_learner learned-investigation --config configs/learned-investigation.json --output runs/learned-investigation-v1`.
All 1680 local per-world/policy/seed rows were checked against every aggregate and
paired difference, query ceilings, distinct validation functions, and clean source
metadata without rerunning worlds. Raw rows and weights remain in ignored `runs/`.
The compact report contains synthetic metrics only. C1 reports/protocol are unchanged.
