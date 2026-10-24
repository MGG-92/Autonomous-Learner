"""Professor wrapper boundaries without pretrained weights or network access."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def test_professor_loads_only_pinned_offline_model_and_freezes_it(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "neural-volume-lab"))
    import worker
    from embeddings import REVISION
    calls = []

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **options):
            assert messages[-1] == {"role": "user", "content": "synthetic greeting"}
            return "templated synthetic greeting"

        def __call__(self, text, **options):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, **options):
            assert ids.tolist() == [3]
            return "synthetic proposal"

    class Model:
        def eval(self): return self
        def requires_grad_(self, enabled):
            assert not enabled
            calls.append("frozen")

        def generate(self, **options):
            assert options["do_sample"] is False
            assert options["max_new_tokens"] == 16
            assert not torch.is_grad_enabled()
            return torch.tensor([[1, 2, 3]])

    class Loader:
        def __init__(self, instance): self.instance = instance

        def from_pretrained(self, name, **options):
            assert options["local_files_only"] is True
            assert options["trust_remote_code"] is False
            assert options["revision"] == REVISION
            assert str(options["cache_dir"]).endswith(str(Path("runs") / "models"))
            calls.append("load")
            return self.instance

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Loader(Tokenizer()), AutoModelForCausalLM=Loader(Model())))
    professor = worker.Professor()
    assert professor.respond("synthetic greeting") == "synthetic proposal"
    assert professor.respond("synthetic greeting") == "synthetic proposal"
    assert calls == ["load", "load", "frozen"]
    with pytest.raises(ValueError):
        professor.respond(" ")
