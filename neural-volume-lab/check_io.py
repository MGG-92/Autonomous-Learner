"""Dense learned-I/O development probe, with frozen and spatial controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from checkpoint import BrainStore
from embeddings import EmbeddingStore
from neural import Config, Learner


def run(output):
    torch.set_num_threads(2)
    store = EmbeddingStore()
    pairs = [(store.encode(a), store.encode(b)) for a, b in [("Hello", "Welcome"), ("Goodbye", "Farewell")]]
    conditions = {}
    checkpoint_result = None
    for name in ("learned_io", "frozen_recurrent_core", "fixed_features_readout_only", "spatial_baseline"):
        config = Config() if name != "spatial_baseline" else Config(encoding="spatial", vocabulary=128)
        learner = Learner(config)
        if name in ("frozen_recurrent_core", "fixed_features_readout_only"):
            learner.network.edges.requires_grad_(False)
        if name == "fixed_features_readout_only":
            for layer in (learner.network.input_adapter, learner.network.input_to_core, learner.network.encoder):
                layer.requires_grad_(False)
        initial = [store.tokenizer.decode(learner.predict(message)) for message, _ in pairs]
        losses = []
        started = time.perf_counter()
        for message, target in pairs:
            learner.admit(message, target)
            losses.append(learner.train(20))
        seconds = time.perf_counter() - started
        predictions = [store.tokenizer.decode(learner.predict(message)) for message, _ in pairs]
        conditions[name] = {"config": learner.description(), "untrained_predictions": initial,
            "predictions": predictions, "exact": sum(prediction == target["text"] for prediction, (_, target) in zip(predictions, pairs)),
            "training": losses, "training_seconds": seconds}
        if name == "learned_io":
            source = store.metadata()
            identity = {k: source[k] for k in ("model", "revision", "dimensions", "vocabulary_size")}
            brain = BrainStore(output / "private-checkpoint" / "brain.pt", identity)
            before = learner.predict(pairs[0][0])
            brain.save(learner)
            fresh = Learner()
            brain.restore(fresh)
            checkpoint_result = {"restored": True, "same_prediction": fresh.predict(pairs[0][0]) == before,
                "same_updates": fresh.updates == learner.updates, "same_vocabulary": fresh.vocabulary == learner.vocabulary,
                "size_MiB": brain.path.stat().st_size / 1024 ** 2}
            brain.close()
    first = pairs[0][0]["tokens"][0]
    permuted = dict(first, vector=[v for p in reversed(first["points"]) for v in [x * store.scale for x in p]],
                    points=list(reversed(first["points"])))
    dense, spatial = Learner(), Learner(Config(encoding="spatial", vocabulary=128))
    with torch.no_grad():
        a = dense.network.tick(torch.zeros_like(dense.network.activity), dense.network.stimulus(first))
        b = dense.network.tick(torch.zeros_like(dense.network.activity), dense.network.stimulus(permuted))
    probe = {"spatial_patterns_identical": torch.equal(spatial.network.stimulus(first), spatial.network.stimulus(permuted)),
             "learned_core_states_differ": not torch.equal(a, b), "dense_first_tick_active_neurons": int((a > 1e-5).sum()),
             "note": "Constructed coordinate-group reversal, not a semantic generalization test"}
    lab = Path(__file__).resolve().parent
    return {"status": "development diagnostic", "seed": 42, "device": "cpu", "threads": 2,
        "runtime": {"torch": torch.__version__}, "epochs_per_pair": 20, "pairs": [["Hello", "Welcome"], ["Goodbye", "Farewell"]],
        "conditions": conditions, "checkpoint": checkpoint_result, "alias_probe": probe,
        "source_sha256": {name: hashlib.sha256((lab / name).read_bytes()).hexdigest() for name in ("neural.py", "checkpoint.py", "worker.py")},
        "limits": ["Known public synthetic development pairs only; not held-out evaluation or evidence of general reasoning.",
            "Frozen/readout controls test functional necessity on this tiny task; no general architecture superiority claim.",
            "The learned 576-to-512 bottleneck and normalization still lose information; they remove the coordinate-stimulation rule, not every limitation.",
            "Checkpoint contains private numeric replay and optimizer state and remains under ignored runs; only this sanitized report is publishable."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.output)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: row["exact"] for name, row in result["conditions"].items()}))
    print(json.dumps({"checkpoint": result["checkpoint"], "alias_probe": result["alias_probe"]}))
