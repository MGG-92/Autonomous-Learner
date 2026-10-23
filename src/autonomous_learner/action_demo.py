"""Scripted development fixture demonstrating belief updates, not policy learning."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from .action_model import TransitionModel
from .action_sampling import Mechanism, instantiate_development
from .action_world import Effect, mechanism_identity


def run(output: Path):
    if output.exists():
        raise FileExistsError("Preserve earlier demos; use a fresh output file")
    effects = (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))
    mechanism = Mechanism(mechanism_identity(effects), effects)
    world = instantiate_development(mechanism, 71, 11, 31, budget=5)
    # Sequence and probe are public slot positions, fixed without inspecting effects.
    # Slot permutations deliberately mean this is not guaranteed to reveal all rules.
    catalog = world.observe().catalog
    model = TransitionModel(catalog)
    probe_initial, probe_actions = (0, 1, 0), (catalog.actions[2], catalog.actions[2])
    before = model.message(probe_initial, probe_actions)
    trace = []
    for index in (0, 0, 1, 2, 2):
        transition = world.act(catalog.actions[index])
        model.observe(transition)
        trace.append(asdict(transition))
    learned = model.message(probe_initial, probe_actions)
    counts = {action: len(candidates) for action, candidates in model.candidates.items()}
    model.reset()
    assert not model.observations and model.simulated_effects == 0
    reset = model.message(probe_initial, probe_actions)
    assert reset == before
    report = {"kind": "scripted_software_demo_not_benchmark", "phase": "development_only",
              "source": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())},
              "settings": {"naming_seed": 71, "permutation_seed": 11, "initial_seed": 31,
                           "action_slots": [0, 0, 1, 2, 2], "budget": 5},
              "trace": trace, "before": before, "after_experience": learned, "after_reset": reset,
              "candidate_counts_after_experience": counts, "real_actions": len(trace),
              "remaining_actions": world.remaining,
              "limitations": ["Named software fixture; no scored prediction or goal test",
                              "Action sequence is scripted, not learned investigation",
                              "Supplied effect templates, uniform factorized prior, filtering and rollout inference",
                              "No parameter training, abstraction acquisition, planning or transfer result"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
