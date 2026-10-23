"""Experience-updated transition beliefs with supplied hypotheses and inference."""
from collections import Counter
from itertools import product

from .action_world import Catalog, STATES, Transition, effect_space, validate_state


class TransitionModel:
    """Independent uniform effect priors, conditioned on observed transitions.

    The prior is over labeled action effects, NOT uniform over canonical worlds in
    the evaluator's population. Probabilities are model-relative, not calibrated
    population frequencies. No weights, primitives or inference algorithm are learned.
    """
    def __init__(self, catalog):
        if not isinstance(catalog, Catalog):
            raise ValueError("The model takes only a public catalog")
        self.catalog = catalog
        self.reset()

    def reset(self):
        self.candidates = {action: effect_space() for action in self.catalog.actions}
        self.observations = []
        self.simulated_effects = 0

    @property
    def valid(self):
        return all(self.candidates.values())

    def observe(self, transition):
        if not isinstance(transition, Transition) or transition.before.catalog != self.catalog:
            raise ValueError("Use a transition from this model's public catalog")
        # The full next state is checked, including properties that did not change.
        old = self.candidates[transition.action]
        remaining = tuple(effect for effect in old
                          if effect.predict(transition.before.values) == transition.after.values)
        self.simulated_effects += len(old)
        self.candidates[transition.action] = remaining
        self.observations.append(transition)

    def distribution(self, initial, actions):
        validate_state(initial)
        if (not isinstance(actions, tuple) or len(actions) > 3
                or any(action not in self.catalog.actions for action in actions)):
            raise ValueError("Predict a tuple of at most three public actions")
        if not self.valid:
            return None  # Explicit global model-class failure; no invented posterior.
        unique = tuple(dict.fromkeys(actions))
        counts, total = Counter(), 0
        # Draw each unknown action effect once per hypothetical world, not per step.
        # Conditioning factorizes across actions because observed states are complete.
        for effects in product(*(self.candidates[action] for action in unique)):
            assignment = dict(zip(unique, effects))
            values = initial
            for action in actions:
                values = assignment[action].predict(values)
                self.simulated_effects += 1
            counts[values] += 1
            total += 1
        return tuple(counts[state] / total for state in STATES)

    def message(self, initial, actions):
        probabilities = self.distribution(initial, actions)
        return {"initial": initial, "actions": actions, "probabilities": probabilities,
                "status": "PREDICTION" if probabilities is not None else "MODEL_CLASS_FAILURE",
                "candidate_counts": {action: len(effects) for action, effects in self.candidates.items()},
                "uncertainty_scope": "supplied independent uniform effect prior; population calibration untested"}
