"""Experimental residual token scorer; not yet evidence of better language output."""
import torch
from torch import nn

from .language import CandidatePage


class TokenController(nn.Module):
    def __init__(self, embedding_dim: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(embedding_dim + state_dim, hidden_dim),
                                      nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.special = nn.Linear(state_dim, 2)  # MORE, END
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.special.weight)
        nn.init.constant_(self.special.bias, -20.0)

    def forward(self, page: CandidatePage, state: torch.Tensor, allow_more: bool = True,
                allow_end: bool = True) -> torch.Tensor:
        if state.ndim != 1:
            raise ValueError("state must be a vector")
        features = torch.cat([page.embeddings, state.expand(len(page.token_ids), -1)], dim=-1)
        token_scores = page.log_probabilities + self.residual(features).squeeze(-1)
        specials = self.special(state)
        mask = torch.tensor([allow_more, allow_end], dtype=torch.bool, device=specials.device)
        specials = specials.masked_fill(~mask, float("-inf"))
        return torch.cat([token_scores, specials])

    def supervised_loss(self, page: CandidatePage, state: torch.Tensor, target_id: int | None,
                        has_more: bool, allow_end: bool = True):
        """Missing visible targets supervise MORE; None supervises END.

        Caller advances the cached page after MORE and supplies the same target.
        Targets beyond the configured maximum must fail, not silently truncate.
        """
        count = len(page.token_ids)
        if target_id is None:
            if not allow_end:
                raise ValueError("END target is masked")
            index = count + 1
        else:
            matches = (page.token_ids == target_id).nonzero().flatten()
            if matches.numel():
                index = int(matches[0])
            elif has_more:
                index = count
            else:
                raise ValueError("Target is unavailable within the candidate budget")
        return nn.functional.cross_entropy(self(page, state, has_more, allow_end)[None],
                                            torch.tensor([index]))


def controlled_generate(engine, controller: TokenController, state: torch.Tensor, prompt: str,
                        max_new_tokens: int = 16, page_size: int = 200, max_pages: int = 2,
                        min_new_tokens: int = 1) -> dict:
    """The controller owns token/MORE/END choices; the LM only supplies candidates.

    Pages accumulate at one fixed next-token distribution. Token IDs are appended
    directly so byte-level/BPE decoding cannot corrupt the next model context.
    """
    if max_new_tokens < 1 or min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("Invalid generation length bounds")
    input_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids
    generated, trace, page_requests = [], [], 0
    for step in range(max_new_tokens):
        cached = engine.candidates_ids(input_ids, page_size=page_size, max_pages=max_pages)
        visible = cached.page(0)
        page_index = 0
        while True:
            with torch.no_grad():
                index = int(controller(visible, state, page_index + 1 < cached.pages,
                                       step >= min_new_tokens).argmax())
            count = len(visible.token_ids)
            if index == count:
                page_index += 1
                page_requests += 1
                next_page = cached.page(page_index)
                visible = CandidatePage(torch.cat([visible.token_ids, next_page.token_ids]),
                            torch.cat([visible.log_probabilities, next_page.log_probabilities]),
                            torch.cat([visible.embeddings, next_page.embeddings]), visible.text + next_page.text)
                trace.append({"action": "MORE", "step": step, "page": page_index})
                continue
            if index == count + 1:
                trace.append({"action": "END", "step": step})
                return {"text": engine.tokenizer.decode(generated), "trace": trace, "page_requests": page_requests}
            token = int(visible.token_ids[index])
            trace.append({"action": "TOKEN", "step": step, "token_id": token,
                          "text": visible.text[index], "log_probability": float(visible.log_probabilities[index])})
            if token == engine.tokenizer.eos_token_id:
                return {"text": engine.tokenizer.decode(generated), "trace": trace, "page_requests": page_requests}
            generated.append(token)
            input_ids = torch.cat([input_ids, torch.tensor([[token]], dtype=input_ids.dtype)], dim=1)
            break
    return {"text": engine.tokenizer.decode(generated), "trace": trace, "page_requests": page_requests}
