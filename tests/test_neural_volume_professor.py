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


@pytest.mark.parametrize("difference,reward", [(5., 1), (-5., -1), (0., 0)])
def test_professor_vote_is_from_frozen_logits_and_explicitly_unverified(monkeypatch, difference, reward):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "neural-volume-lab"))
    import worker
    professor = worker.Professor()
    monkeypatch.setattr(professor, "respond", lambda message: "A short example")

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert '"learner_reply": "candidate"' in messages[0]["content"]
            return "synthetic question"

        def __call__(self, prompt, **options):
            return {"input_ids": torch.tensor([[1]])}

        def encode(self, label, **options):
            return [2 if label == "Yes" else 3]

    class Model:
        def __call__(self, **options):
            assert not torch.is_grad_enabled()
            logits = torch.zeros(1, 1, 4)
            logits[0, 0, 2] = difference
            return SimpleNamespace(logits=logits)

    professor.tokenizer, professor.model = Tokenizer(), Model()
    result = professor.assess("synthetic question", "candidate")
    assert result["reward"] == reward
    assert result["source"].startswith("Unverified")
    assert "not calibrated" in result["source"]
