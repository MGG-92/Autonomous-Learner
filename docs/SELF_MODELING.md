# Functional self-modeling and metacognition: research direction

Status: long-term objective and proposed experiments, not an implemented capability.
Keep the world-model, investigation and goal-planning milestones. First establish
the supplied planner baseline; then introduce a small reliability-learning pilot.
Self-modeling does not need to wait for language rendering or general abstraction
discovery, and it must not replace those research questions.

## Operational objective and limits

Learn predictions about the agent's own knowledge, attainable performance, costs
and the consequences of its cognitive operations from experienced outcomes. Use
those predictions to allocate effort, investigate, revise or switch strategies.
The target is a useful predictive/control model, potentially implicit and nonverbal,
not a stored autobiography or convincing first-person output. Functional success
does not establish subjective consciousness; these experiments do not adjudicate it.

An explicit separate self-model is a hypothesis, not a required architecture.
A single controller with history may adapt equally well. Simple confidence
calibration may suffice for a narrow task without representing capabilities.
Claim only the distinctions supported by controls; retaining parameters/history
or speaking about uncertainty is insufficient. No learner can infer an invisible
change that has no observable consequence; identify what evidence makes an
internal limitation distinguishable from a difficult or changed external world.
If the task model is already correct and its controller optimal under the available
resources, a separate monitor may add only overhead. Do not manufacture an advantage
by withholding resources or observations from the no-self-model control.
Here a no-self-model baseline means no designated learned self-predictor/state;
a history-based controller could still acquire an implicit equivalent. Equal benefit
would argue against the separate module's necessity, not prove an absence of
functional metacognition. Missing retrieval is observable; the identity or truth of
forgotten facts is not automatically available to introspection.

Some priors must be supplied: observable interfaces, possible operations, feedback
semantics and resource constraints. Supplying access to a memory count is not
supplying the learned conclusion that missing evidence makes a strategy unreliable.
Conversely, hardcoding that conclusion or an error threshold is a heuristic baseline,
not discovery of self-knowledge. Preserve both the raw-feedback and feature priors
in every report. Do not require learning without any representation bias.

## Existing support and genuinely new mechanisms

| Existing component | What it supports | What it does not establish |
|---|---|---|
| Rule/action posterior distributions and candidate counts | Observable uncertainty features and predictions to compare with later experience | Knowledge of blind spots, calibrated reliability or awareness of uncertainty |
| Actual transition records and bounded-memory/reset controls | Evidence of prediction errors and controlled loss of accessible information | Learned diagnosis of why information is missing or whether a strategy will fail |
| Supplied investigation rules and C2 trainable query scorer | Comparable cognitive operations; the mixed training result is an important control | A learned model of strategy reliability or a demonstrated training benefit |
| Memory controllers and documented timing failures | Future tasks involving resource constraints and retention decisions | General introspection or knowledge of limitations; research reports are not learner evidence |
| Effect-evaluation counters and explicit budgets | Costs that can be exposed through a declared, priced interface | An ability to predict costs or choose computation effectively |
| Structured messages / optional language adapter | A later way to express committed internal predictions | Self-model learning, semantic fidelity or consciousness |

New work is needed to define legitimate internal observations; predict one's own
future errors, memory accessibility and strategy outcomes; update those predictions
after actual feedback; and learn/use control choices that improve net task utility.
None of these is demonstrated by the current world-prediction scores.

## Learning principles and information boundary

External model: predict next world state from observed state and action.
Candidate self-model: predict accessible memory, error, success and cost from
observable belief/context, chosen strategy, resource constraints and cognitive action.
Both can use hypotheses, committed predictions, feedback, revision and interventions.
The current 45 guarded-effect templates are not a suitable universal cognitive
model class: self dynamics may be noisy, delayed and nonstationary as the agent learns.
Begin with a small empirical predictor; compare alternative representations later.

Define a minimal agent-observable event interface only when the pilot is specified.
Examples: its committed prediction, selected strategy, accessible evidence summary,
retrieval success/failure, actual computation spent, and ordinary subsequent task
feedback. Distinguish directly read counters from inferred capabilities. The task
model and self-model may share learned representations; no mandatory second agent,
language narrator or recursive hierarchy is required.

Research audit logs, private conversations, held-out panel targets, true mechanisms,
oracle reachability, intervention labels and evaluator error diagnoses remain
inaccessible. Current offline prediction panels cannot simply become self-training
data. Training feedback must be deliberately available ordinary experience; held-out
evaluation stays evaluator-only. Predict before feedback. Commit models/selection
and declare any permitted online updating before evaluation; never tune on final data.

A self-model's history and parameters consume declared memory/compute budgets.
Do not restore forgotten task facts from a monitoring log or unbounded side store.
Specify whether an intervention resets task memory alone or all learned state;
ablate both where relevant. Bounded access to aggregated experience may be allowed,
but the retention mechanism and remaining information must be explicit.

## Staged experiments (all pending)

1. **C5a: reliability forecasting after a goal baseline.** Freeze a task model and
   planner first. Learn a small predictor of that agent's future error or goal
   success from ordinary development episodes. Compare with its raw task-model
   probabilities, constant/base-rate forecasts, a scalar calibrator, task-only
   predictors and fixed uncertainty thresholds. Use proper scores, calibration and
   error discrimination separately from task performance. Forecasting alone is
   a monitoring result, not improved control or a full self-model.
2. **C5b: agent-specific capability and process effects.** Independently vary task
   difficulty and the agent's accessible evidence, memory capacity, strategy or
   computation budget. Randomize/match interventions in development trials to
   distinguish causal effects from selection bias: a strategy chosen for hard tasks
   can fail more often while still being better. Do not expose hidden condition
   IDs or canned explanations. Predict consequences using observable experience;
   test new mechanism/constraint combinations and adaptation after unannounced shifts.
   Compare task-only features with added internal context, shuffled/stale histories
   and another agent's matched history. An agent-specific gain must survive task
   difficulty/base-rate controls; uncertainty may remain irreducible.
3. **C5c: metacognitive control.** Let the agent choose a bounded extra experiment,
   retrieval, reconsideration, strategy change or stopping decision. Compare against
   the same task learner without the self-model, matched-information fixed heuristics,
   an equally sized history-based controller and a monitor-only system whose outputs
   cannot affect behavior. Remove/shuffle the learned self-model while preserving
   the task model to test whether it causes the benefit. Match feedback and charge
   monitoring, memory, training amortization and extra actions. Report net goal
   success/cost, not narrative quality or confidence alone.
4. **C5d: improving learning strategies.** After a C5c benefit, learn proposals,
   reusable strategies or update rules using performance feedback on development
   tasks. Connect to C2 search learning and C3 abstractions. Compare with from-scratch,
   exact initialization, fixed-strategy and no-self-model controls, including failed
   updates. Hold out task families and capability configurations, not just names.
   Changes to learning are parameter/algorithm learning only when actually trained
   or induced; supplied switching among fixed strategies remains supplied machinery.

The first pilot should remain CPU-friendly: a small structured success forecaster,
one task family, one declared internal constraint, frozen planning, and no language
or autonomous code rewriting. Specify sample sizes, resource limits, feedback timing,
seeds and development/validation/final partitions in a new protocol before running.
These are candidate designs, not a preregistered completed experiment.

## Failure criteria and conceptual obstacles

No benefit over a simple calibrator/task-only predictor weakens the case for a
distinct self-model. Better forecasts without better resource-adjusted decisions
support monitoring only. Gains explained by extra memory, feedback, compute, task
identities or intervention flags do not establish the proposed capability. A fluent
description with unchanged decisions is not evidence. Test nonverbal operation and
language removal before using a renderer to communicate self-related claims.

Main obstacles: separating self limitations from world uncertainty, learning about
unchosen strategies without counterfactual gold, nonstationarity and delayed credit,
feedback/selective-observation bias, monitoring overhead, and representing procedural
knowledge beyond a fixed vocabulary. These motivate controlled steps rather than a
universal self-awareness module. Evidence for one constraint/task is not general
self-understanding, open-ended self-improvement or subjective consciousness.

## Research grounding

- [Fleming & Lau (2014), How to measure metacognition](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00443/full)
  is a measurement review distinguishing confidence bias, sensitivity and efficiency;
  confidence level or correctness alone is not a sufficient metacognitive measure.
  Human-task measures must not be transferred uncritically to this agent.
- [Bongard, Zykov & Lipson's self-modeling work, author laboratory](https://www.creativemachineslab.com/evolutionary-self-modeling.html)
  demonstrates experience-grounded models of robot body dynamics used for behavior.
  It motivates intervention/adaptation tests, not a claim that body modeling proves
  cognitive introspection or validates our architecture.
- [Wang et al., Learning to reinforcement learn](https://arxiv.org/abs/1611.05763)
  provides a concrete learned-adaptation approach. Meta-learning can be implemented
  through recurrent dynamics without a separately named self-model; our proposed
  explicit model must earn its complexity against such history-based alternatives.

These sources motivate tests and distinctions; the staged design above is a project
proposal. No result in them is evidence that Autonomous Learner already has these abilities.
