# Interactive Agent Playground — planned

An accessible way to give the agent goals, watch it investigate artificial worlds,
and inspect the actual model behind its predictions. This is an observability and
manual experimentation tool, not evidence of improved reasoning. No playground is
implemented yet. The intended command is `python -m autonomous_learner playground`;
it is a future interface, not a currently available command.

## Place in the research workflow

Finish the first scored development action-world goal baseline before implementing
P1 in `docs/PLAN.md`. The simulator, transition model and exhaustive planning
primitive already exist; the goal runner and scored controls remain pending.
Then take one bounded CLI increment that exposes these components. Do not wait for
natural-language understanding, general abstractions, self-modeling or a graphical
interface. Keep the C5a reliability pilot and C3 research moving; interface polish
must not become a prerequisite for their experiments.

The earlier M2.5 memory/activity viewer remains a later graphical extension. Its
historical graph-memory integration dependencies do not block the action-world CLI.
Reuse a shared event format rather than build competing explanations of the agent.

## P1: smallest useful version

- Start a fresh, explicitly development-only sandbox with the existing objects,
  binary properties and unknown guarded action effects. Show legal actions, public
  observations, budgets, configured controller and model assumptions.
- Accept structured goals and prediction/inspection queries. Initially these are
  constrained commands over public object/action identifiers, not unrestricted
  natural-language conversation. Unsupported questions return an explicit limitation.
- Investigate with the actual selected controller; inspect the observations, remaining
  effect candidates, predictions and evidence supporting candidate elimination.
  State what remains ambiguous. Model-relative probabilities are not demonstrated
  empirical calibration or learned knowledge of the agent's reliability.
- Plan through the existing supplied planner over acquired dynamics; show the
  committed sequence, predicted success and simulated work, then execute that
  sequence within its declared budget. No hidden reachable-goal hint or silent
  replanning. Distinguish an empty plan, exhausted budget and model-class failure.
- Provide step and run modes, inspection of current state, explicit reset/new-session
  commands and safe exit. Reset must say which world, evidence, controller state
  and budgets it replaces. Do not silently restore forgotten information.

Label each operation by its source: observations acquired from experience; belief
updates within supplied templates; programmed investigation/search/inference;
trained parameters only when a real trained component is loaded. Record its identity
and training provenance. Optional LLM parsing/rendering is a later, separately labeled
adapter; it must not supply hypotheses or choose investigations in the controlled core.
The existing scripted `action-model-demo` is not this interactive agent.

## Execution and extension boundary

Use narrow adapters to `action_world.py`, `action_model.py`,
`action_investigation.py` and `action_planning.py`, with the same execution path for
step/run modes. The investigation loop currently runs a whole budget at once;
extracting resumable execution will require meaningful parity checks, not a second
implementation of action selection. The goal runner should be reused once available.

Emit versioned, ordered events from actual operations: public observation, candidate
decision and recorded scores where available, chosen action, actual transition,
belief change, prediction, plan commitment, execution and budget/work changes.
Include session/model/controller identity and evidence IDs. Step advances one declared
operation boundary; it need not expose every arithmetic instruction. Inspection and
replay render captured state without rerunning stochastic choices, querying the
oracle or incrementing simulation counters. Additional predictions requested by the
user are explicit computations with separately recorded work, not free inspection.
Missing telemetry is unknown, not an invented explanation or chain of thought.

Observers may retain a local historical trace, but that trace never becomes an agent
input. Show current accessible memory separately from history. Connect the existing
bounded-memory facilities only when actually integrated into this action-world path;
do not imply that the current transition model already has learned memory management.
Future adapters can expose acquired abstractions, reliability forecasts and cognitive
control when implemented. They must use a declared agent-visible internal-event
interface, not researcher audit logs or forgotten task contents.

## Safe manual experimentation

Keep manual sessions in ignored `runs/playground/`, separate from benchmark outputs,
selection, training and checkpoints. Do not automatically publish human interactions
or ingest sessions as training data. Show sandbox provenance and qualitative status.
Use the existing development mechanism partition, not validation/final loaders or
benchmark target panels. Different names or random seeds alone do not isolate worlds:
mechanisms are equivalent under object/action renaming. No reserved-world import.

P1 need not offer environmental rule editing. Later add explicitly logged operator
interventions and unfamiliar development situations, with public observations as the
agent's only feedback. Within the current finite family, check canonical mechanism
identity against reserved partitions before admitting edits/imports. For richer custom
families, establish an exclusion registry/protocol before allowing them; do not claim
isolation from seed labels alone. Manual sessions and mechanisms cannot subsequently
be promoted into a reserved evaluation set.

Do not silently reveal a rule, reset beliefs or signal a hidden change to make
adaptation look successful. The current stationary fixed-template model can exhaust
its candidates after a change; report `MODEL_CLASS_FAILURE` honestly. Structural
revision, noise handling and changing-world adaptation require new research. Omit
hidden-rule reveal from P1; any later operator-only view must stay outside agent inputs.

## Acceptance gates (future checks, not results)

- Match core actions, observations, predictions and budgets under identical world
  and controller seeds with recording off/on and step/run. Inspecting/replaying
  consumes no oracle calls, changes no RNG state and preserves recorded work counts.
- Trace actual evidence changes and ambiguity; charge real no-op actions once and
  distinguish requested prediction work from recorded decision work. Verify a committed
  plan executes without receiving new feedback during its original selection.
- Refuse reserved imports and cross-session leakage; keep observer history inaccessible
  after reset/forgetting. Test any later rule editing against intervention logging,
  partition checks and honest model-class failure before advertising adaptation.
- Document a short fresh-session walkthrough: inspect prior, step investigation,
  inspect evidence/prediction, set goal, plan and execute. Use real components and
  label supplied algorithms. Manual success is qualitative, not a benchmark result.

All P1 gates are pending. A GUI, replay scrubbing, arbitrary language, rule editing,
persistent-session recovery and new cognitive capabilities are separate later work.
