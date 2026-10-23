# Frozen memory robustness — preliminary negative results

An experiment developed primarily by agentic AI. Evaluation source
`ebd209995ed0cd6565fdd5923f60baae754185e1`, clean working tree; original training
source `d675f4b38749c16b7054644b1ca846beed90d946`. The
[protocol](../../docs/experiments/delayed-memory-robustness.md) and world reservation
were committed before evaluation. No retraining, checkpoint selection, reward tuning,
or new pass/fail gate was performed. Checkpoint hashes and selections are in report.json.

The three frozen optimizer seeds 7/42/97 share 100 fresh worlds, 2100000..2100099,
across eight scenarios. This is 100 unique worlds, not 2400 independent worlds.
Each world has nine related possible questions. All percentages below average the
three optimizer seeds; utility subtracts 0.01 per consolidation write.

| Scenario | Learned accuracy | Scheduled graph accuracy | FIFO accuracy | Learned minus graph utility, 95% world bootstrap |
|---|---:|---:|---:|---|
| Reference | 22.11% | 19.56% | 6.11% | +0.0179 [-0.0073, +0.0435] |
| Reverse order | 21.85% | 19.89% | 3.89% | +0.0117 [-0.0162, +0.0376] |
| Two-tick observation gaps | 0.00% | 0.00% | 0.00% | +0.0300 [+0.0300, +0.0300] |
| Ten final quiet ticks | 18.67% | 5.67% | 5.67% | +0.1101 [+0.0919, +0.1277] |
| Three belief slots | 7.44% | 7.00% | 0.11% | -0.0006 [-0.0113, +0.0106] |
| Nine belief slots | 36.63% | 26.11% | 18.78% | +0.0965 [+0.0656, +0.1281] |
| Three allowed writes | 6.15% | 1.00% | 1.00% | +0.0415 [+0.0267, +0.0567] |
| Nine allowed writes | 24.59% | 28.33% | 4.67% | -0.0291 [-0.0424, -0.0162] |

## What failed and what the numbers mean

All three learned controllers and their imitation ablations made **zero writes**
with observation gaps, achieving zero answers. Random retained enough evidence for
4.19% mean accuracy. The positive learned-minus-graph utility in this scenario is
only saved write cost: both learned and scheduled graph answer nothing. It is not
successful retention. Gaps leave at most one episodic observation visible at a time
before any consolidation; the policies trained on adjacent observations do not
successfully acquire beliefs here. This result identifies timing sensitivity, not
a proven causal explanation of particular network features.

With nine allowed writes, learned mean utility is 0.1742 versus scheduled graph
0.2033. More budget does not establish superiority: the learned policy uses 7.17
writes on average, versus graph's 8, and still answers fewer questions. With only
three belief slots there is no clear utility advantage over graph. Even the fresh
reference comparison's interval includes zero; retain the original narrower result
without extending it to a robust advantage over the strong baseline.

Nine belief slots improve learned accuracy, but the storage ceiling is larger than
in the original experiment. The ten-tick quiet tail changes public elapsed-time
features and consolidation scheduling during observation; the scheduled graph uses
only three writes instead of five. Beliefs have no decay, so this is not evidence
of long-term biological-style retention or learned forgetting over time. Reversal
of a shuffled stream preserves its marginal distribution and tests paired order
sensitivity rather than a new graph distribution.

All operational policies obey hard storage/write limits: **zero capacity violations**.
Untrained and no-memory accuracy are zero throughout. Dynamic graph, random,
imitation-only, and all per-seed metrics remain in the full report and CSVs.
Inference, graph family, and features are supplied; retrieval and revision are not learned.

Intervals use 2000 resamples of worlds, averaging optimizer seeds within world,
bootstrap seed 197. They are exploratory and unadjusted across comparisons, and
do not measure uncertainty across optimizer seeds. Shared worlds also correlate
scenarios. These results do not complete M2.

## Reproduction and next work

Run the command in the protocol using final-v1 checkpoints, or first reproduce
training from the original frozen source and configuration. Checkpoints remain
local in ignored runs. The evaluation took 25.27 seconds on this development CPU;
this is not a portable performance benchmark. report.json contains source and
environment metadata. Per-world metrics are split by scenario into eight CSVs
for publication; they contain the same rows as the local combined output.

These worlds are now OPENED and must not be used for tuning. Broader timing/resource
training needs separate training/validation ranges and another reserved final range.
The next engineering increment integrates the frozen delayed-memory controller into
the persistent investigation loop, with these limitations explicit, before M2.5.
