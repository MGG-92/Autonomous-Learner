# C2 development experiment: learned query ranking

This is a bounded development comparison, not a final evaluation. C1's source,
protocol, reports, and opened final functions remain unchanged. This command has
no final phase. A future final study needs a fresh protocol and unopened functions
or world families; a new input layout of an opened function is not a fresh rule.

## Question and supplied machinery

Can a shared query scorer learn to choose useful observations from the resulting
prediction improvement, without entropy labels or expert query demonstrations?
The potentially transferable idea is ranking experiments by current model
disagreement. Transfer to action worlds is not tested here. Sorted predictive
masses remove query coordinates and output-category names; that invariance is
supplied, not a learned abstraction. Five outcomes remain a fixed dimensional limit.

The same 1210-function C1 public hypothesis class supplies exact version-space
filtering and uniform predictive distributions. Every model-based scorer scans
every remaining legal query. The learned scorer is a 7 -> 16 -> 1 tanh MLP, given
five sorted probabilities, normalized log model count, and remaining budget fraction.
It receives no hidden program, target values, program identity, coordinates, entropy
targets, pretrained weights, language, or evaluator traces.

## Training and separation

Train on the C1 development function partition only. Training layout seeds begin
at 4000000, validation at 4100000; ranges must be disjoint. Per episode, choose one
development function using data seed 411 and reserve five prediction probes, never
legal as queries. The reward is prior minus final multiclass Brier loss on those
training probes. Gold targets are training-only reward labels, not policy features.
Use REINFORCE with summed action log probabilities, exponential reward baseline
(0.95 old + 0.05 current, updated after the loss), and summed entropy bonus 0.01.
Adam learning rate 0.005, 800 episodes, query ceiling four, optimizer seeds 7/42/97.
Use one CPU thread (the CLI sets this); save the last episode, no checkpoint or
hyperparameter selection on validation. Training data order/layouts are shared
across seeds; initialization and stochastic training actions differ.

Validation samples 80 distinct functions from C1's validation partition using seed
521; it is development evidence and may inform future protocols. Layout/query RNG
seed is independent of the hidden rule. Pair worlds and reserved probes across all
policies. Evaluate frozen scorers greedily, breaking score ties with a public RNG.
Report optimizer seeds separately. No confidence interval treating shared worlds or
seed repetitions as independent samples is claimed in this preliminary comparison.

## Controls, cost, and falsification

- Same seed's exact untrained network, information gain, random, and fixed query order.
- Prior-feature control: query ranking sees the unconditioned model after every
  observation; predictions still use the actual learned posterior. This tests
  whether adaptive model access helps beyond a fixed prior-based query preference.
- Post-query model reset: remove posterior restrictions and raw observations before
  prediction. A no-memory predictor discarding the same replies is equivalent;
  these are one control, not independent evidence.

All controls share interaction ceilings and stop at unique identification. Count
selection and total hypothetical output reads, plus query wall time and training
time/read cost. Selection reads count simulated candidate outputs; total also counts
filtering and prediction. Prior-feature scans are counted in addition to the actual
learner. Catalog construction and neural/optimizer arithmetic are excluded from read
counts. Wall time includes scoring overhead but is descriptive, not a portable
benchmark or equal compute budget. Training cost is separate; do not claim compute
superiority without a future matched-budget/amortization study.

Score predictions before reading validation gold targets, using accuracy, Brier,
and log loss. Report paired per-world Brier differences and individual seed results.
Failure to beat initialization/random, loss against information gain, or no benefit
over prior features is informative. Better arithmetic predictions would demonstrate
query-control learning under this supplied representation, not learned inference,
new concepts, causal discovery, or general reasoning. Preserve negative outcomes;
move toward the structured action-world bridge rather than tuning this indefinitely.

## Reproduction

`python -m autonomous_learner learned-investigation --config configs/learned-investigation.json --output runs/learned-investigation-v1`

Use the project venv and a fresh output directory. Weights/raw outputs stay in ignored
`runs/`. Publish only a reviewed compact synthetic report, with source hash, dirty
status, all config/seeds, individual outcomes, cost limits, and interpretation.
