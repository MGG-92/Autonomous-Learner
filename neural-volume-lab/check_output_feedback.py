"""Seeded development smoke; feedback connectivity is not dialogue competence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from embeddings import EmbeddingStore
from neural import Config, Learner


def run():
    torch.set_num_threads(2)
    store = EmbeddingStore()
    pairs = [(store.encode(a), store.encode(b)) for a,b in
             [("Hello world", "Welcome"), ("Goodbye world", "Farewell")]]
    conditions = {}
    for enabled in (False, True):
        learner = Learner(Config(output_feedback=enabled))
        started = time.perf_counter()
        training = []
        for message, target in pairs:
            learner.admit(message, target)
            training.append(learner.train_online(12))
        predictions = [store.tokenizer.decode(learner.predict(message)) for message,_ in pairs]
        net = learner.network
        with torch.no_grad():
            state = torch.rand(net.activity.shape, generator=torch.Generator().manual_seed(93)) * 4
            peak = float(state.max())
            ratios = []
            for _ in range(60):
                after = net.tick(state)
                ratios.append(float(after.max() / state.max().clamp(min=1e-30)))
                state = after
            causal = None
            if enabled:
                logits, actual = net(learner.encode(pairs[0][0]))
                weights = net.output_to_core.weight.clone()
                net.output_to_core.weight.zero_()
                ablated_logits, ablated = net(learner.encode(pairs[0][0]))
                net.output_to_core.weight.copy_(weights)
                causal = {"nonzero_weights": int(weights.count_nonzero()),
                    "positive_weights": int((weights > 0).sum()), "negative_weights": int((weights < 0).sum()),
                    "maximum_row_L1": float(weights.abs().sum(-1).max()),
                    "same_input_state_max_difference_after_feedback_ablation": float((actual-ablated).abs().max()),
                    "same_input_logits_max_difference_after_feedback_ablation": float((logits-ablated_logits).abs().max())}
        conditions["feedback" if enabled else "no_feedback"] = {
            "description": learner.description(), "training": training, "predictions": predictions,
            "exact": sum(p==t["text"] for p,(_,t) in zip(predictions,pairs)),
            "undriven_peak_before": peak, "undriven_peak_after_60_ticks": float(state.max()),
            "maximum_undriven_peak_ratio": max(ratios), "causal_probe": causal,
            "seconds": time.perf_counter()-started}
    lab = Path(__file__).resolve().parent
    return {"status":"development smoke", "seed":42, "perturbation_seed":93, "threads":2,
        "torch":torch.__version__, "device":"cpu", "updates_per_pair":12,
        "pairs":[["Hello world","Welcome"],["Goodbye world","Farewell"]],
        "conditions":conditions,
        "source_sha256":{name:hashlib.sha256((lab/name).read_bytes()).hexdigest()
            for name in ("neural.py","checkpoint.py","worker.py","check_output_feedback.py")},
        "limits":["Known development pairs, no held-out evaluation or checkpoint selection.",
            "The feedback ablation tests an effect on internal computation, not improved task performance.",
            "Matched update counts and shared initial existing weights; added feedback parameters change capacity.",
            "No evidence of general conversation, general reasoning or learned automatic stopping after a delivered message.",
            "Feedback uses previous tick's continuous output units, not sampled text tokens or professor output."]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result=run()
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:{"exact":value["exact"],"causal_probe":value["causal_probe"]}
        for name,value in result["conditions"].items()}))
