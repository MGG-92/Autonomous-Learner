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
| Internal rule discovery v1 | No parameter training; development semantic hash buckets 0..69 + named reference functions | Semantic buckets 70..84; layout 3200000..3200099 | OPENED: 100 sampled rules from semantic buckets 85..99; layout 3300000..3300099 |

The delayed-memory protocol originally reserved 900000..900199, then moved the range
before any final run to avoid the preexisting demo. Different optimizer seeds share
the same evaluation worlds; do not multiply the unique-world count by optimizer seeds.

The rule-discovery generator is separate from the graph-world generator. Whole
finite-domain function signatures are disjoint across its partitions; query seeds
7/42/97 are random/tie repeats, not training seeds. See its versioned protocol.
Named primitive/max/projection reference functions are always development-only,
overriding their hash bucket; software examples cannot become final-rule evidence.
