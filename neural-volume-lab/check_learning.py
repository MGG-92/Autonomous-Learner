"""Seeded development smoke with controls; synthetic associations, not a benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from embeddings import EmbeddingStore
from neural import Config, Learner


def run():
    torch.set_num_threads(2)
    store = EmbeddingStore()
    pairs = [(store.encode(a), store.encode(b)) for a, b in [("Hello", "Welcome"), ("Goodbye", "Farewell")]]
    rows = {}
    for name in ("learned_replay", "frozen_recurrent_core", "no_replay", "no_input", "untrained"):
        learner = Learner()
        if name == "frozen_recurrent_core":
            learner.network.edges.requires_grad_(False)
        edges = learner.network.edges.detach().clone()
        training = []
        for message, target in pairs:
            if name == "no_replay":
                learner.examples.clear()
            learner.admit(message, target)
            if name == "no_input":
                pulses, labels = learner.examples[-1]
                learner.examples[-1] = (torch.zeros_like(pulses), labels)
            if name != "untrained":
                training.append(learner.train(100))
        outputs = []
        for message, target in pairs:
            if name == "no_input":
                with torch.no_grad():
                    logits, _ = learner.network(torch.zeros_like(learner.encode(message)))
                indices = logits[:, :len(learner.vocabulary)+1].argmax(-1).tolist()
                ids = []
                for index in indices[:learner.config.slots]:
                    if index == 0: break
                    ids.append(learner.vocabulary[index - 1])
            else:
                ids = learner.predict(message)
            outputs.append({"input": message["text"], "expected": target["text"],
                            "prediction": store.tokenizer.decode(ids), "exact": ids == [t["id"] for t in target["tokens"]]})
        rows[name] = {"outputs": outputs, "exact": sum(p["exact"] for p in outputs), "total": len(outputs),
                      "training": training, "core_parameter_l1_change": float((learner.network.edges.detach() - edges).abs().sum())}
    random = np.random.default_rng(42)
    support = [0] + list(dict.fromkeys(t["id"] for _, target in pairs for t in target["tokens"]))
    random_outputs = []
    for message, target in pairs:
        ids = []
        for _ in range(16):
            token = int(random.choice(support))
            if token == 0: break
            ids.append(token)
        random_outputs.append({"input": message["text"], "prediction": store.tokenizer.decode(ids),
                               "exact": ids == [t["id"] for t in target["tokens"]]})
    rows["random_readout"] = {"outputs": random_outputs, "exact": sum(p["exact"] for p in random_outputs), "total": 2}
    lab = Path(__file__).resolve().parent
    digests = {name: hashlib.sha256((lab / name).read_bytes()).hexdigest()
               for name in ("neural.py", "embeddings.py")}
    return {"status": "development smoke", "seed": 42, "architecture": Learner().description(),
            "code_sha256": digests, "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": 2},
            "optimizer": {"name": "Adam", "lr": .003, "epochs_per_pair": 100, "gradient_clip": 1.,
                          "task_loss": "token cross entropy through first EOS; no target spatial loss"},
            "embedding": store.metadata(), "results": rows,
            "limits": ["Two public synthetic taught pairs, no reserved evaluation data or checkpoint selection.",
                "These are software/memorization controls, not estimates of generalization.",
                "A frozen-core result tests whether recurrent parameter learning was necessary for this task.",
                "No pretrained language generation participates in learner predictions."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh report path")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: f"{row['exact']}/{row['total']}" for name, row in result["results"].items()}))
