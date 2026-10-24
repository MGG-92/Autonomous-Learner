"""Offline JSON-lines compute process. All text and learned state stay in RAM."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


class Professor:
    def __init__(self):
        self.model = self.tokenizer = None

    def respond(self, message):
        if not isinstance(message, str) or not message.strip() or len(message) > 4096:
            raise ValueError("Professor needs 1–4096 characters")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from embeddings import MODEL_ID, REVISION
        if self.model is None:
            cache = ROOT / "runs" / "models"
            options = dict(cache_dir=str(cache), revision=REVISION, local_files_only=True, trust_remote_code=False)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_safetensors=True,
                dtype=torch.float32, **options).eval()
            self.model.requires_grad_(False)
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": "Reply briefly, in at most one short sentence."},
            {"role": "user", "content": message}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Engine:
    def __init__(self):
        import torch
        torch.set_num_threads(2)
        from embeddings import EmbeddingStore
        from neural import Learner
        self.store = EmbeddingStore()
        self.store.load()
        self.learner = Learner()
        self.professor = Professor()

    def activity(self):
        state = self.learner.network.activity.detach()
        ids = (state > 1e-5).nonzero().flatten()
        return {"active": int(ids.numel()), "peak": float(state.max()),
                "activity": [[int(i), float(state[i])] for i in ids], "updates": self.learner.updates}

    def handle(self, request):
        import torch
        op = request["op"]
        if op == "teacher":
            return {"proposal": self.professor.respond(request["message"]),
                    "source": "Frozen local SmolLM2-135M; unverified proposal"}
        if op == "init":
            return self.learner.description() | self.activity()
        if op == "admit":
            self.learner.admit(self.store.encode(request["message"]), self.store.encode(request["target"]))
            return self.learner.description()
        if op == "train":
            return self.learner.train(request.get("epochs", 10))
        if op == "rest":
            return self.learner.rest(request.get("epochs", 10))
        if op == "predict":
            ids = self.learner.predict(self.store.encode(request["message"]))
            return {"prediction": self.store.tokenizer.decode(ids), "ids": ids} | self.activity()
        if op == "step":
            with torch.no_grad():
                pulse = None
                if request.get("message"):
                    sequence = self.store.encode(request["message"])
                    pulse = self.learner.network.pulse(sequence["tokens"][request.get("index", 0) % len(sequence["tokens"])]["points"])
                self.learner.network.activity.copy_(self.learner.network.tick(self.learner.network.activity, pulse))
            return self.activity()
        if op == "silence":
            self.learner.network.activity.zero_()
            return self.activity()
        if op == "reset":
            from neural import Learner
            self.learner = Learner()
            return self.learner.description() | self.activity()
        raise ValueError("Unknown operation")


def main():
    engine = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if engine is None:
                engine = Engine()
            result = engine.handle(request)
            print(json.dumps({"op": request["op"], "result": result}), flush=True)
        except Exception as error:
            # No private prompt/text traceback is written to disk.
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
