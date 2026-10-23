import math
import time

import torch

from .agent import Agent, run_episode
from .memory import Memory
from .policy import Heuristic, RandomPolicy
from .world import SyntheticTeacher, World


def evaluate(policy, start: int, worlds: int, max_steps: int = 8, capacity: int = 64) -> dict:
    if worlds < 1:
        raise ValueError("worlds must be positive")
    results = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for seed in range(start, start + worlds):
            world = World.generate(seed)
            memory = Memory(capacity=capacity)
            agent = Agent(memory, world.catalog(), world.goal(), max_steps)
            result, _ = run_episode(agent, SyntheticTeacher(world), policy, world.truth(world.goal()))
            results.append(result)
            memory.close()
    n = len(results)
    p = sum(r.correct for r in results) / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    answered = sum(r.answered for r in results)
    return {
        "worlds": n, "accuracy": p, "accuracy_wilson95": [max(0.0, center - half), min(1.0, center + half)],
        "coverage": answered / n,
        "selective_accuracy": sum(r.correct for r in results) / answered if answered else None,
        "mean_reward": sum(r.reward for r in results) / n,
        "mean_questions": sum(r.questions for r in results) / n,
        "mean_steps": sum(r.steps for r in results) / n,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def compare(policy, initial, config) -> dict:
    settings = (config.test_start, config.test_worlds, config.max_steps)
    return {
        "learned": evaluate(policy, *settings),
        "untrained": evaluate(initial, *settings),
        "random": evaluate(RandomPolicy(config.seed), *settings),
        "heuristic": evaluate(Heuristic(), *settings),
        "learned_no_memory": evaluate(policy, *settings, capacity=0),
    }
