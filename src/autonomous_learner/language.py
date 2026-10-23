"""Optional frozen local language engine. No imports/downloads until explicitly loaded."""
from dataclasses import dataclass
from pathlib import Path

import torch

from .types import Answer, Query

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass
class CandidatePage:
    token_ids: torch.Tensor
    log_probabilities: torch.Tensor
    embeddings: torch.Tensor
    text: list[str]


class CachedCandidates:
    def __init__(self, logits: torch.Tensor, embeddings, decode, page_size: int = 200,
                 max_pages: int | None = None, temperature: float = 1.0):
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Expected finite one-dimensional vocabulary logits")
        if page_size < 1 or (max_pages is not None and max_pages < 1) or temperature <= 0:
            raise ValueError("Candidate limits and temperature must be positive")
        self.log_probs = (logits.detach().float().cpu() / temperature).log_softmax(-1)
        self.ranking = torch.argsort(self.log_probs, descending=True, stable=True)
        self.page_size = page_size
        pages = (len(logits) + page_size - 1) // page_size
        self.pages = min(pages, max_pages) if max_pages else pages
        self.embeddings, self.decode = embeddings, decode

    def page(self, index: int) -> CandidatePage:
        if index < 0 or index >= self.pages:
            raise IndexError("Candidate page unavailable")
        ids = self.ranking[index * self.page_size:(index + 1) * self.page_size]
        # Probabilities retain normalization over the ENTIRE vocabulary.
        return CandidatePage(ids, self.log_probs[ids], self.embeddings(ids).detach().cpu(),
                             [self.decode(int(i)) for i in ids])


class FrozenLanguageModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache: Path = Path("runs/models"),
                 device: str = "cpu", local_only: bool = False, revision: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device = model_name, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache),
                            trust_remote_code=False, local_files_only=local_only, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache),
                         trust_remote_code=False, use_safetensors=True,
                         local_files_only=local_only, revision=revision, dtype=torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        self.forward_calls = 0

    def candidates(self, context: str, **kwargs) -> CachedCandidates:
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        if inputs.input_ids.numel() == 0:
            raise ValueError("Context must contain at least one token")
        return self.candidates_ids(inputs.input_ids, **kwargs)

    def candidates_ids(self, input_ids: torch.Tensor, **kwargs) -> CachedCandidates:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.numel():
            raise ValueError("Expected one nonempty token sequence")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to(self.device), use_cache=False)
        self.forward_calls += 1
        embedding = self.model.get_input_embeddings()
        return CachedCandidates(outputs.logits[0, -1],
                    lambda ids: embedding(ids.to(self.device)),
                    lambda token_id: self.tokenizer.decode([token_id]), **kwargs)

    def complete(self, prompt: str, max_new_tokens: int = 48) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        context = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def template_answer(goal: Query, answer: Answer) -> str:
    if answer.status == "SUPPORTED":
        return f"{goal.subject} can {goal.object}."
    if answer.status == "REFUTED":
        return f"{goal.subject} cannot {goal.object}."
    if answer.status == "CONFLICT":
        return f"I have conflicting evidence about whether {goal.subject} can {goal.object}."
    return f"I do not know whether {goal.subject} can {goal.object}."


def render_answer(engine: FrozenLanguageModel, goal: Query, answer: Answer) -> dict:
    # Synthetic entity names are copied AFTER generation, never reconstructed by
    # selecting unlikely name tokens. Strict acceptance is an intentionally narrow
    # fidelity baseline, not a general semantic-equivalence checker.
    skeleton = template_answer(Query("SUBJECT", goal.relation, "OBJECT"), answer)
    proposal = engine.complete(f"Repeat this sentence exactly, without explanation: {skeleton}")
    accepted = proposal.strip() == skeleton
    return {"text": proposal.replace("SUBJECT", goal.subject).replace("OBJECT", goal.object)
            if accepted else template_answer(goal, answer),
            "raw_proposal": proposal, "accepted": accepted, "fallback": not accepted,
            "evidence_ids": answer.evidence_ids}
