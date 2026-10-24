# Action-world goals: first development comparison

Source/config/protocol freeze: `47aee8d1b81aad5a4e745c84df2447536549cb08`, clean tree.
See the [protocol](../../docs/experiments/action-goals.md) and
[compact synthetic report](development.json). Prior reports remain unchanged.

**Limited result:** experience-conditioned planning has higher observed success
than prior-only planning, but its paired world interval crosses zero. Information
gain does not outperform random or fixed investigation. Only 10 of 80 sampled goals
are truly reachable within three actions. Do not claim reliable improvement over
the prior or general goal-directed reasoning from this measurement.

## Design and results

Use the existing 80 development mechanisms, 12 sequential exploration actions and
controller seeds 7/42/97. No optimizer, LLM or new primitive is involved. All policies
share one independently sampled goal-start/goal pair per world. Goals differ from
starts and are sampled without inspecting reachability. All policy/seed commitments
are saved before any goal feedback or evaluator reachability analysis. Execute an
open-loop sequence of at most three actions without model updates or replanning.

Information-gain/random/fixed exploration each condition the same supplied effect
model and feed the same exhaustive planner. Reset/no-memory/untrained uses the
erased information-gain model. Random-plan ignores the model and samples three
independent action slots uniformly. Both controls reuse IG exploration costs; they
are not extra independent investigation runs. Shorter model plans win probability
ties; random plans always use the full ceiling.

Success below means exact final goal state. Primary rates include ALL 80 tasks.
Means weight worlds equally, then average controller seeds. All seeds share worlds
and goals; the 240 executions per policy are not 240 independent tasks.

| Investigation / planning | Successes / 80, seed 7 | Seed 42 | Seed 97 | Mean all-goal success | Mean reachable-goal success | Mean goal actions |
|---|---:|---:|---:|---:|---:|---:|
| Information gain + updated model | 9 | 8 | 8 | 10.42% | 83.33% | 1.5792 |
| Random + updated model | 8 | 9 | 9 | 10.83% | 86.67% | 1.6958 |
| Fixed cycle + updated model | 9 | 8 | 8 | 10.42% | 83.33% | 1.7000 |
| Reset / no memory / untrained prior | 6 | 5 | 5 | 6.67% | 53.33% | 3.0000 |
| Random full-budget plan | 3 | 3 | 2 | 3.33% | 26.67% | 3.0000 |

Reachable-goal rates use only the same **10 worlds per seed**; 70 goals are
unreachable and every policy fails on those. The all-goal ceiling on this sampled
panel is 12.5%. This ceiling is computed afterward and never enters a decision.
The small reachable subset limits precision; it must not replace primary results.

Paired all-goal success gains for IG, averaging seeds within world before 2000
world bootstrap resamples (seed 197). Intervals are exploratory/unadjusted, conditional
on these development tasks; values below are percentage points.

| Comparator | Mean gain | World bootstrap 95% interval |
|---|---:|---:|
| Random investigation | -0.42 | [-1.25, 0.00] |
| Fixed investigation | 0.00 | [0.00, 0.00] |
| Reset / prior planning | +3.75 | [-1.25, 9.17] |
| Random plan | +7.08 | [1.67, 12.92] |

Identical sampled world-mean success versus fixed does not establish policy
equivalence. The prior comparison remains inconclusive despite its higher point
estimate. The random-plan comparison supports a bounded benefit of this model/search
system over that control, not learned planning or superiority of information gain.

## Resource use and verification

| Policy | Mean exploration effect evaluations | Mean planning effect evaluations | Mean goal no-ops |
|---|---:|---:|---:|
| Information gain | 1043.12 | 129660.60 | 0.9833 |
| Random investigation | 293.58 | 137116.20 | 1.0458 |
| Fixed investigation | 291.28 | 132979.75 | 1.0083 |
| Reset / prior | 1043.12 (reused) | 1774710.00 | 1.9583 |
| Random plan | 1043.12 (reused) | 0 | 2.2042 |

Every actual investigation uses 12 actions, including no-ops. There are 720 real
investigation runs / 8640 actions and 1200 goal executions / 2634 actual goal actions.
Goal action ceilings match, actual lengths differ, and computation is not matched.
All model planners enumerate 40 sequences. Filtering can substantially reduce
exhaustive search work; this is a consequence of supplied enumeration over fewer
candidates, not acquisition of a new reasoning strategy. Wall times are descriptive:
the complete run took 483.96 seconds on Python 3.12.14.

139 tests and fresh legacy/action-model/goal smoke demos passed. Independent offline
checks reconstructed all 1200 score rows and 1200 cost rows from saved commitments
and transitions, checked goals and canonical development identity, verified true
reachability by enumerating all sequences, and reconstructed summaries/bootstrap
intervals. Verification required no new environment interactions. Raw sessions remain
in ignored `runs/`; the public report contains synthetic aggregate evidence only.

## Limits and next questions

This is one goal per previously opened development mechanism, within a small supplied
stationary effect vocabulary. No final set, mechanism-family transfer, acquired
abstraction, trained planner, online revision, calibrated reliability or self-model
was tested. Success probabilities remain relative to the supplied prior. Neither
long-term autonomy nor subjective consciousness follows from these results.

Preserve the negative findings and this sparse goal coverage. Any broader goal
panel needs its own declared protocol before measurement; do not select reachable
goals after seeing truth or tune on these outcomes. The bounded P1 playground can
now expose the actual model and its failures. C5a needs a legitimate internal-feedback
schema and matched controls, rather than training on researcher-only evaluation logs.
