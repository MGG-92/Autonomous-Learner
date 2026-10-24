"""Local static embeddings and transparent coordinate triples; no generation."""
import math
import hashlib
import json
from pathlib import Path
import sys
import threading


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MAX_TOKENS = 64


def coordinate_triples(vector, scale):
    if not vector or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Use a nonempty vector and a positive finite scale")
    if any(not math.isfinite(v) for v in vector):
        raise ValueError("Embedding components must be finite")
    padded = list(vector) + [0.0] * ((-len(vector)) % 3)
    return [[v / scale for v in padded[i:i + 3]] for i in range(0, len(padded), 3)]


class LocalTokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.engine = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self.engine.encode(text, add_special_tokens=add_special_tokens).ids

    def convert_ids_to_tokens(self, token_id):
        return self.engine.id_to_token(token_id)

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return self.engine.decode(ids, skip_special_tokens=False)


class EmbeddingStore:
    def __init__(self, cache=None):
        bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "model"
        development = Path(__file__).resolve().parent.parent / "runs" / "neural-volume-lab" / "model"
        self.cache = Path(cache) if cache else (bundled if bundled.exists() else development)
        self._lock = threading.Lock()
        self.tokenizer = self.weights = None

    def load(self):
        with self._lock:
            if self.weights is not None:
                return
            import numpy as np
            try:
                manifest = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
                if manifest["model"] != MODEL_ID or manifest["revision"] != REVISION:
                    raise ValueError("Unexpected embedding source")
                for name in ("embeddings.npy", "tokenizer.json"):
                    with (self.cache / name).open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                    if digest != manifest["sha256"][name]:
                        raise ValueError("Local model integrity mismatch")
                weights = np.load(self.cache / "embeddings.npy", mmap_mode="r", allow_pickle=False)
                if weights.shape != (49152, 576) or weights.dtype != np.float32:
                    raise ValueError("Unexpected embedding table")
                tokenizer = LocalTokenizer(self.cache / "tokenizer.json")
                scale = float(manifest["scale"])
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("Embedding table has no valid nonzero scale")
            except Exception as error:
                raise RuntimeError("Local embedding model unavailable. See the setup instructions in neural-volume-lab/README.md.") from error
            self.scale = scale
            self.tokenizer, self.weights = tokenizer, weights

    def metadata(self):
        self.load()
        return {"model": MODEL_ID, "revision": REVISION, "kind": "static_input_embeddings",
                "dimensions": self.weights.shape[1], "vocabulary_size": self.weights.shape[0],
                "scale": self.scale, "scaling": "each component / maximum absolute component in the entire frozen table",
                "max_tokens": MAX_TOKENS, "teacher": False, "neural_simulation": False}

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a nonempty message")
        if len(text) > 4096:
            raise ValueError("Use at most 4096 characters")
        self.load()
        # No chat template, BOS/EOS insertion, hidden states or model forward pass.
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > MAX_TOKENS:
            raise ValueError(f"Use between one and {MAX_TOKENS} tokens; this text has {len(ids)}")
        tokens = []
        for token_id in ids:
            vector = self.weights[token_id].tolist()
            tokens.append({"id": token_id, "piece": self.tokenizer.convert_ids_to_tokens(token_id),
                           "text": self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                           "vector": vector, "points": coordinate_triples(vector, self.scale),
                           "padded_components": (-len(vector)) % 3})
        return {"text": text, "tokens": tokens}
