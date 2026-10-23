"""Supplied investigation controls: policy inputs contain no evaluator state."""
import math
import random
import time

from .action_world import Observation, Transition


def choose_action(model, current, strategy, rng, step):
    if not isinstance(current, Observation) or current.catalog != model.catalog or not model.valid:
        raise ValueError("Choose from a valid model and its public observation")
    actions = model.catalog.actions
    if strategy == "fixed":
        return actions[step % len(actions)]
    if strategy == "random":
        return rng.choice(actions)
    if strategy != "information_gain":
        raise ValueError("Unknown investigation control")
    # Deterministic effects: next-state entropy equals expected model information gain.
    entropies = []
    for action in actions:
        probabilities = model.distribution(current.values, (action,))
        entropies.append(-sum(p * math.log(p) for p in probabilities if p))
    best = max(entropies)
    return rng.choice([action for action, entropy in zip(actions, entropies) if abs(entropy - best) < 1e-12])


def investigate(model, initial, act, strategy, seed, budget):
    if type(budget) is not int or budget < 1:
        raise ValueError("Use a positive interaction budget")
    rng, current = random.Random(seed), initial
    selection_work, selection_seconds = 0, 0.0
    began, work_before = time.perf_counter(), model.simulated_effects
    observation_start = len(model.observations)
    for step in range(budget):
        before, start = model.simulated_effects, time.perf_counter()
        action = choose_action(model, current, strategy, rng, step)
        selection_seconds += time.perf_counter() - start
        selection_work += model.simulated_effects - before
        transition = act(action)  # The sole real-world contact in this loop.
        if (not isinstance(transition, Transition) or transition.before != current
                or transition.action != action):
            raise ValueError("Reply must match the chosen action and current observation")
        model.observe(transition)
        if not model.valid:
            raise ValueError("In-class development world exhausted the model")
        current = transition.after
    return {"real_actions": budget, "selection_effect_evaluations": selection_work,
            "exploration_effect_evaluations": model.simulated_effects - work_before,
            "selection_seconds": selection_seconds, "exploration_seconds": time.perf_counter() - began,
            "noop_actions": sum(t.before.values == t.after.values for t in model.observations[observation_start:])}
