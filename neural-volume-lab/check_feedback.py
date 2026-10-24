"""Seeded policy-update diagnostic, not a dialogue or generalization benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neural import Config, Learner


def sequence(ids):
    return {"tokens": [{"id": i, "points": [[0, 0, 0]]} for i in ids]}


def run():
    torch.set_num_threads(2)
    rows = []
    cases = [("human_positive", 1, "human", 5), ("human_negative", -1, "human", 5),
             ("human_weak", 1, "human", 1), ("professor_positive", 1, "professor", 1),
             ("no_reward", 0, "human", 5), ("cleared_replay_positive", 1, "human", 5)]
    for seed in (7, 42, 97):
        for name, reward, source, strength in cases:
            model = Learner(Config(seed=seed, encoding="spatial", vocabulary=128))
            model.admit(sequence([1]), sequence([8, 9]))
            ids, trajectory = model.sample(sequence([1]))
            if name == "cleared_replay_positive":
                model.examples.clear()
                model.example_metadata.clear()
            result = model.feedback(trajectory, reward, source, strength)
            rows.append({"condition": name, "seed": seed, "sampled_token_ids": ids,
                         "sampled_actions_including_EOS": trajectory["classes"], **result,
                         "delta_mean_log_probability": result["mean_log_probability_after"] - result["mean_log_probability_before"]})
    # A frozen uniform random policy is unaffected by rewards by construction.
    random = np.random.default_rng(42)
    baseline = {"kind": "fixed uniform random readout; no optimizer", "seed": 42,
                "sampled_actions": random.integers(0, 3, 16).tolist(), "delta_mean_log_probability": 0.}
    lab = Path(__file__).resolve().parent
    return {"status": "development mechanism diagnostic", "seeds": [7, 42, 97],
            "config": Learner(Config(encoding="spatial", vocabulary=128)).description(), "rows": rows, "random_baseline": baseline,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__, "threads": 2, "device": "cpu"},
            "source_sha256": {name: hashlib.sha256((lab / name).read_bytes()).hexdigest() for name in ("neural.py", "worker.py")},
            "hyperparameters": {"feedback_SGD_base_lr": .05, "human_default_strength": 5,
                "professor_strength": 1, "professor_reward_weight": .25, "gradient_clip": 1.,
                "per_token_ratio_clip": [.8, 1.2], "feedback_updates_per_source_per_reply": 1},
            "limits": ["Immediate scalar-reward mechanism only, not long-horizon RL or dialogue quality.",
                "Admitted target tokens provide equal readout support; no supervised optimization precedes measurement.",
                "No pretrained model, held-out task, checkpoint selection or private dialogue is involved.",
                "Clipped updates can saturate; no guaranteed large change or improvement in correctness."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh report path")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: [round(row["delta_mean_log_probability"], 6) for row in result["rows"] if row["condition"] == name]
                      for name in dict.fromkeys(row["condition"] for row in result["rows"])}))
