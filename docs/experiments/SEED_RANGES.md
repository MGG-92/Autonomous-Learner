# Synthetic-world seed ledger

Fresh seeds change names/content, not the fixed generator's overall graph family.
Do not call a previously inspected range an untouched confirmatory dataset. Training
and validation may be revisited; freeze final tests before examining results.

| Experiment | Training | Validation | Reported/final evaluation |
|---|---|---|---|
| Original question controller | 0..319 | 100000..100039 | Development 200000..200099; replication 500000..500199 |
| Original hinted memory | 0..79 | 100000..100009 | Development/reproduction 200000..200099 |
| Original integrated controllers | Models above | Models above | 600000..600199 |
| Delayed-specific-goal memory | 700000..700359 | 800000..800039 | Reserved 2000000..2000199 |
| Frozen delayed-memory robustness v1 | No updates; original models above | No selection | OPENED 2100000..2100099; shared across eight scenarios |
| Delayed-memory unit/smoke | 1100000 and nearby | 1200000 and nearby | 1300000 and nearby; software checks only |
| Interactive original demo | No parameter updates | Not applicable | 900001 (and user-selected seeds) |

The delayed-memory protocol originally reserved 900000..900199, then moved the range
before any final run to avoid the preexisting demo. Different optimizer seeds share
the same evaluation worlds; do not multiply the unique-world count by optimizer seeds.
