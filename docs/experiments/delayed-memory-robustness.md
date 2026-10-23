# Frozen delayed-memory robustness v1

Freeze this protocol and evaluation code before opening worlds 2100000..2100099.
Evaluate the three validation-selected final-v1 checkpoints and their imitation-only
ablations from source `d675f4b38749c16b7054644b1ca846beed90d946`. Do not retrain,
select checkpoints, alter rewards, or tune on these worlds. Record checkpoint SHA-256,
selection metadata, training config, and evaluation source in the report. Checkpoints
remain local in ignored runs; reproduction can retrain the original frozen protocol.

Use the same 100 worlds for all scenarios and optimizer seeds 7/42/97. Baselines:
untrained initialization, random, scheduled FIFO, dynamic graph, scheduled graph,
and learned with zero belief capacity. Episodic evidence must expire before scoring.
Inference and available public task vocabulary remain supplied and unchanged.

| Scenario | Change from training configuration |
|---|---|
| reference | None; fresh worlds |
| reverse_order | Reverse the generated shuffled fact sequence |
| observation_gaps | Two empty ticks between observations; horizon 45 ticks |
| long_quiet_delay | Ten final empty ticks instead of two; horizon 25 ticks |
| small_belief | Three belief slots instead of six |
| large_belief | Nine belief slots instead of six |
| few_writes | Three consolidation operations instead of six |
| many_writes | Nine consolidation operations instead of six |

Each observation-gap scenario has 15 facts, 14 two-tick gaps, and two final ticks.
The public horizon always reflects the full stream. Two episodic slots/TTL are fixed;
gaps therefore expire each observation before the next arrives. Gap ticks also offer
decision opportunities: this is a combined timing/horizon shift. Reversing a uniformly
shuffled stream preserves its marginal distribution while testing paired order
sensitivity. A longer quiet tail does not test decay of permanent belief storage.

Report accuracy, coverage, utility (accuracy minus 0.01 per write), actual writes,
and hard capacity violations per seed/scenario/policy, plus synthetic per-world rows.
Report learned-minus-FIFO, scheduled-graph, and imitation utility differences using
the existing 2000-resample world bootstrap, seed 197, after averaging optimizer seeds
within world. All intervals are exploratory, unadjusted across scenarios/comparisons,
and do not quantify optimizer-seed uncertainty. No new pass/fail gate is claimed.

Do not change final-v1 claims based on this evaluation. If shifts fail, broaden a
separate training/validation distribution and reserve another final range before
testing that new policy. Persistent-loop integration and the M2.5 viewer follow;
joint training, learned retrieval, and revision remain unfinished.

```powershell
.\.venv\Scripts\python.exe -m autonomous_learner delayed-robustness --checkpoints runs/delayed-final-v1 --output runs/delayed-robustness-v1
```
