# Delayed-goal memory experiment — protocol v1

Status: defined before implementing/training this experiment. Treat as a preliminary
agentic-AI experiment. Do not tune against the final test split or replace failed runs.

## Question

Can an optimized policy learn when to consolidate and which evidence to retain when
the specific future question is withheld and observations are randomly interleaved?
The previous memory benchmark supplied the future goal and placed its proof first.

## Environment and information boundary

- Reuse the random synthetic world: 3 entities, 6 classes, 3 capabilities, 15 facts.
- Shuffle ALL facts, with no ordering by their usefulness to any future question.
- The public task specification supplies entity/capability vocabulary and the finite
  horizon, but no selected future question, hidden facts, stream suffix, or seed.
- Only after the stream and a short quiet delay, evaluate all 9 entity/capability
  queries. Averaging these is expected accuracy for a uniform delayed question.
- Inference rules remain supplied. This tests memory control, not learning logic.
- Policy inputs are computed exclusively from the bounded accessible stores and
  public task specification. Training may use expert labels and terminal rewards.

## Resource budgets

- Episodic store: at most 2 facts, with a 2-tick lifetime, including quiet ticks.
- Belief store: at most 6 facts. No extra feature history, hidden archive, or log replay.
- At most 6 consolidation operations in 17 ticks (15 observations + 2 quiet ticks).
- Each tick chooses WAIT or CONSOLIDATE. Consolidation selects up to 6 visible facts
  from the union of episodic and belief stores. Removed evidence cannot be recovered
  unless it is observed again. Invalid writes (no new evidence/budget exhausted) are masked.
- Terminal utility = fraction of delayed questions answered correctly minus 0.01
  per consolidation. Unknown answers count as unanswered, not confident errors.

## Policies and training

Learn a shared neural evidence encoder, retention scores, and WAIT/CONSOLIDATE gate.
Bootstrap from a goal-free graph heuristic on training worlds, then run policy-gradient
fine-tuning on separate training worlds. Expert selection may search bounded visible
subsets; its logic is supplied, and imitation must be reported honestly. Validation
utility alone selects checkpoints, with earlier checkpoints winning ties.

Compare untrained, random, evenly scheduled FIFO, dynamic graph heuristic, and evenly
scheduled graph heuristic. Every operational policy has the SAME hard memory/write
limits. Also report no-memory, unlimited-storage reference, and an offline best-subset
bound. The offline bound sees the entire stream and is not an operational baseline.

## Frozen main run

Configuration: `configs/delayed-memory.json`; optimizer seeds 7, 42, 97.
Training seeds start at 700000 (120 imitation worlds + 240 RL worlds); validation
800000..800039; final tests 900000..900199. Splits must be checked for overlap.
Unit-test and smoke configurations use different ranges. Hyperparameters may be
debugged using training/validation only; final reports must include every main seed.

## Evidence and decision gate

Report delayed-query accuracy, selective accuracy/coverage, writes, utility, capacity
violations, actual store sizes, checkpoint stage, learning curves, per-world metrics,
runtime/config/source metadata, and budget-matched baselines. Audit logs are never
policy inputs. Test feature invariance to hidden truth/suffix/selected future query,
real forgetting, quiet expiration, persistence, masks, and bounded resources.

Success requires improvement over initialization in each optimizer seed and a positive
mean paired utility difference against FIFO with a 95% world-resampling interval
excluding zero. Average optimizer seeds within each world before resampling; 3 seeds
are not broad robustness evidence. Matching/beating a stronger graph heuristic is a
separate comparison; do not imply it from beating FIFO. Failure leaves M2 incomplete.

Report the best-subset storage bound so a low absolute accuracy is not misrepresented
as a model failure when the storage budget makes perfect recall impossible. Do not
claim broad memory management, learned retrieval, source reliability, or logic learning.
