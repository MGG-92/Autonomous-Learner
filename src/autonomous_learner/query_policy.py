"""Trainable investigation over public model predictions; no oracle or language input."""
import math

import torch
from torch import nn


class QueryPolicy(nn.Module):
    """Shared scorer, invariant to query ordering and output-category renaming.

    Sorted predictive masses, relative log model count, and remaining query budget
    are supplied features. No coordinates, program IDs, truth, or entropy labels.
    This representation is still an inductive bias, not a learned abstraction.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def query_features(learner, remaining_budget, total_budget):
    if learner.space.modulus != 5 or not 1 <= remaining_budget <= total_budget:
        raise ValueError("This diagnostic uses five outcomes and a positive query budget")
    pairs = tuple(pair for pair in learner.legal_queries if pair not in learner.asked)
    count = math.log1p(len(learner.candidates)) / math.log1p(len(learner.space.hypotheses))
    rows = []
    for pair in pairs:
        probabilities = learner.distribution(pair)
        if probabilities is None:
            raise ValueError("Cannot select queries after model-class failure")
        rows.append([*sorted(probabilities), count, remaining_budget / total_budget])
    return pairs, torch.tensor(rows, dtype=torch.float32).reshape(-1, 7)


def choose_query(policy, learner, rng, remaining_budget, total_budget):
    pairs, features = query_features(learner, remaining_budget, total_budget)
    if not pairs:
        return None
    with torch.no_grad():
        logits = policy(features).tolist()
    best = max(logits)
    return rng.choice([pair for pair, score in zip(pairs, logits) if abs(score - best) < 1e-7])
