import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from eval.validate import check
from rag.answer import GROUNDED, UNGROUNDED, answer, judge, rewrite
from rag.models import LLM, FakeReranker, LLMError, transient

HISTORY = [{"role": "user", "content": "How many sick days do we get?"}, {"role": "assistant", "content": "10 per year."}]


class FakeClient:
    """Answers say how many excerpts they were given ("excerpts=3"), or `text` if set. The judge's verdict
    is `verdict(prompt)`."""

    def __init__(self, text: str | None = None, verdict=lambda prompt: True, stop_reason: str = "end_turn") -> None:
        self.text, self.verdict, self.stop_reason, self.calls = text, verdict, stop_reason, []
        self.messages = self

    def create(self, **request):
        self.calls.append(request)
        prompt = request["messages"][0]["content"]
        if "format" in request["output_config"]:
            text = json.dumps({"correct": self.verdict(prompt), "reason": "r"})
        else:
            text = self.text if self.text is not None else f"excerpts={prompt.count('<excerpt>')}"
        return SimpleNamespace(stop_reason=self.stop_reason, model=request["model"],
                               content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def test_repository_data_is_valid():
    assert check() == []


def test_no_retrieval_route_answers_without_excerpts(tmp_path):
    client = FakeClient()
    llm = LLM(client, tmp_path)
    answer(llm, "What is the capital of Japan?", None, [])
    assert client.calls[0]["system"] == UNGROUNDED and "<excerpt>" not in client.calls[0]["messages"][0]["content"]
    excerpt = {"document_id": "employee-handbook", "section": "Sick leave", "content": "10 paid sick days."}
    assert answer(llm, "And do they carry over?", [excerpt], HISTORY)["text"] == "excerpts=1"
    request = client.calls[1]
    assert request["system"] == GROUNDED and "temperature" not in request and request["output_config"] == {"effort": "low"}
    assert "<conversation>\nuser: How many sick days do we get?\nassistant: 10 per year.\n</conversation>" in \
        request["messages"][0]["content"]


def test_rewrite_and_judge_see_the_conversation(tmp_path):
    client = FakeClient("Do unused sick days carry over to the next year?")
    llm = LLM(client, tmp_path)
    assert rewrite(llm, "And do they carry over?", HISTORY)["text"].startswith("Do unused sick days")
    verdict = judge(llm, "And do they carry over?", "No.", "No, they don't.", HISTORY)
    assert verdict["correct"] and "user: How many sick days" in client.calls[1]["messages"][0]["content"]
    assert client.calls[1]["output_config"]["effort"] == "high"


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_incomplete_responses_raise_with_their_cost_and_are_not_cached(tmp_path, stop_reason):
    client = FakeClient(stop_reason=stop_reason)
    llm = LLM(client, tmp_path)
    for _ in range(2):
        with pytest.raises(LLMError) as raised:
            rewrite(llm, "And then?", HISTORY)
    assert raised.value.input_tokens == 100 and len(client.calls) == 2 and not list(tmp_path.iterdir())


def test_bad_key_or_model_errors_are_not_transient():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(cls, code):
        return cls("x", response=httpx.Response(code, request=request), body=None)
    assert transient(anthropic.APIConnectionError(request=request))
    assert transient(status(anthropic.RateLimitError, 429)) and transient(status(anthropic.InternalServerError, 503))
    assert not transient(status(anthropic.AuthenticationError, 401)) and not transient(status(anthropic.NotFoundError, 404))


def test_fake_reranker_orders_by_query_word_overlap():
    chunks = [{"content": "parking garage"}, {"content": "sick days carry over"}, {"content": "sick leave"}]
    assert [c["content"] for c in FakeReranker().rerank("do sick days carry over", chunks)] == [
        "sick days carry over", "sick leave", "parking garage"]


def test_real_model_client_requires_explicit_model_ids(monkeypatch):
    import pytest
    monkeypatch.delenv("GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    with pytest.raises(ValueError, match="GENERATOR_MODEL"):
        LLM()
