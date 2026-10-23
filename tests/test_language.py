import pytest
import torch

from autonomous_learner.language import CachedCandidates, render_answer, template_answer
from autonomous_learner.teachers import LocalLMTeacher
from autonomous_learner.token_controller import TokenController
from autonomous_learner.types import Answer, Query


def test_fixed_distribution_pagination_and_global_probabilities():
    logits = torch.arange(450, dtype=torch.float32) / 100
    embedding = torch.nn.Embedding(450, 8)
    candidates = CachedCandidates(logits, embedding, str)
    pages = [candidates.page(i) for i in range(3)]
    assert [len(p.token_ids) for p in pages] == [200, 200, 50]
    assert torch.equal(torch.cat([p.token_ids for p in pages]), torch.arange(449, -1, -1))
    assert torch.cat([p.log_probabilities.exp() for p in pages]).sum().item() == pytest.approx(1.0)
    assert torch.equal(candidates.page(0).token_ids, pages[0].token_ids)
    with pytest.raises(IndexError):
        candidates.page(3)
    with pytest.raises(ValueError):
        CachedCandidates(logits, embedding, str, page_size=0)


def test_reranker_can_optimize_target_and_more_end_are_real_actions():
    torch.manual_seed(4)
    embedding = torch.nn.Embedding(6, 6)
    embedding.weight.data.copy_(torch.eye(6))
    candidates = CachedCandidates(torch.arange(6).float(), embedding, str, page_size=3)
    page = candidates.page(0)
    scorer = TokenController(6, 2)
    state = torch.zeros(2)
    assert int(scorer(page, state).argmax()) == 0  # Highest language prior initially.
    optimizer = torch.optim.Adam(scorer.parameters(), lr=0.05)
    before = float(scorer.supervised_loss(page, state, 3, True).detach())
    for _ in range(30):
        loss = scorer.supervised_loss(page, state, 3, True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < before / 10
    assert int(page.token_ids[int(scorer(page, state).argmax())]) == 3
    assert torch.isfinite(scorer.supervised_loss(page, state, 0, True))
    assert torch.isfinite(scorer.supervised_loss(page, state, None, False))
    with pytest.raises(ValueError, match="unavailable"):
        scorer.supervised_loss(page, state, 0, False)
    assert torch.isneginf(scorer(page, state, False, False)[-2:]).all()


class StubEngine:
    def __init__(self, response):
        self.response = response

    def complete(self, prompt):
        return self.response


def test_renderer_copies_names_and_falls_back_on_changed_claim():
    goal, answer = Query("invented_x", "can", "invented_y"), Answer("REFUTED", (1,))
    accepted = render_answer(StubEngine("SUBJECT cannot OBJECT."), goal, answer)
    assert accepted["accepted"] and accepted["text"] == "invented_x cannot invented_y."
    rejected = render_answer(StubEngine("SUBJECT can OBJECT."), goal, answer)
    assert rejected["fallback"] and rejected["text"] == template_answer(goal, answer)


def test_local_teacher_does_not_promote_invalid_or_mismatched_json():
    query = Query("a", "can", "x")
    assert LocalLMTeacher(StubEngine('{"object":"x","positive":true}'), "").ask(query).facts[0].positive
    assert not LocalLMTeacher(StubEngine('{"object":"x","positive":"true"}'), "").ask(query).facts
    assert not LocalLMTeacher(StubEngine('{"object":"wrong","positive":true}'), "").ask(query).facts


def test_generation_can_request_more_then_select_previously_visible_token():
    from types import SimpleNamespace
    from autonomous_learner.token_controller import controlled_generate

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, prompt, return_tensors):
            return SimpleNamespace(input_ids=torch.tensor([[1]]))

        def decode(self, ids):
            return ",".join(map(str, ids))

    class Engine:
        tokenizer = Tokenizer()
        calls = 0

        def candidates_ids(self, ids, **kwargs):
            self.calls += 1
            return CachedCandidates(torch.arange(6).float(), torch.nn.Embedding(6, 2), str, **kwargs)

    class Selection(torch.nn.Module):
        def forward(self, page, state, allow_more, allow_end):
            scores = torch.zeros(len(page.token_ids) + 2)
            # Request page 2, then choose page 1's best token. Already exposed
            # candidates must remain available without a new language-model call.
            scores[len(page.token_ids) if allow_more else 0] = 1
            return scores

    engine = Engine()
    result = controlled_generate(engine, Selection(), torch.zeros(1), "prompt", max_new_tokens=1,
                                 page_size=3, max_pages=2)
    assert result["text"] == "5"
    assert result["page_requests"] == 1 and engine.calls == 1
