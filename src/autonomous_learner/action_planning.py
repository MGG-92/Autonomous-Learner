"""Supplied exhaustive planning over acquired dynamics; no planner learning or self-model."""
from dataclasses import dataclass
from itertools import product

from .action_world import STATES, validate_state


@dataclass(frozen=True)
class Plan:
    actions: tuple[str, ...]
    predicted_success: float | None
    evaluated_sequences: int
    effect_evaluations: int


def plan(model, initial, goal, max_actions, rng):
    """Commit an open-loop plan using only the model and public initial/goal states.

    Maximize predicted success, then prefer a shorter sequence; use the provided
    public RNG for remaining ties. No environment instance, callback, reachable-goal
    hint, evaluator target, or introspective claim enters this function.
    Predicted success is a probability under the supplied task-model prior, not
    an empirically learned estimate of the agent's own reliability.
    """
    validate_state(initial)
    validate_state(goal)
    if type(max_actions) is not int or not 0 <= max_actions <= 3:
        raise ValueError("Use an integer planning budget from zero through three")
    if not model.valid:
        raise ValueError("Cannot plan after model-class failure")
    goal_index = STATES.index(goal)
    before = model.simulated_effects
    best, ties, evaluated = -1.0, [], 0
    for length in range(max_actions + 1):
        for actions in product(model.catalog.actions, repeat=length):
            probability = model.distribution(initial, actions)[goal_index]
            evaluated += 1
            if probability > best + 1e-12:
                best, ties = probability, [(actions, probability)]
            elif abs(probability - best) <= 1e-12 and len(actions) == len(ties[0][0]):
                ties.append((actions, probability))
    chosen, probability = rng.choice(ties)
    return Plan(chosen, probability, evaluated,
                model.simulated_effects - before)
