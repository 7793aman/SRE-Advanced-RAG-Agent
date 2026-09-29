"""Unit seam: ServiceInvoker's own logic (intent gating, tavily gating) — the
actual RAG pipeline call is faked, matching tests/test_rag_service.py's approach
of not touching OpenAI/Qdrant/Redis from a unit test.
"""

from __future__ import annotations

import pytest

from app.models import ChatResponse
from eval.invokers import ServiceInvoker, SkippedIntent


@pytest.fixture
def fake_response() -> ChatResponse:
    return ChatResponse(answer="A Pod is...", sources=["pods.html"], retrieval_score=0.9)


def test_invoke_passes_question_and_flags_through_to_the_real_pipeline(
    monkeypatch: pytest.MonkeyPatch, fake_response: ChatResponse
) -> None:
    calls = []

    def _fake_run_rag_with_trace(question: str, flags: dict) -> tuple[ChatResponse, list]:
        calls.append((question, flags))
        return fake_response, []

    monkeypatch.setattr("eval.invokers.run_rag_with_trace", _fake_run_rag_with_trace)

    resp, chunks = ServiceInvoker().invoke("What is a Pod?", {"search_mode": "dense"}, "rag")

    assert calls == [("What is a Pod?", {"search_mode": "dense"})]
    assert resp is fake_response
    assert chunks == []


def test_invoke_rejects_sql_intent_in_service_mode() -> None:
    with pytest.raises(SkippedIntent, match="sql"):
        ServiceInvoker().invoke("How many pods?", {}, "sql")


def test_invoke_rejects_hybrid_intent_in_service_mode() -> None:
    with pytest.raises(SkippedIntent, match="hybrid"):
        ServiceInvoker().invoke("How many P1 incidents...", {}, "hybrid")


def test_invoke_skips_web_fallback_without_a_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eval.invokers.settings.tavily_api_key", "")
    with pytest.raises(SkippedIntent, match="tavily"):
        ServiceInvoker().invoke("What's the weather?", {}, "web_fallback")


def test_invoke_allows_web_fallback_with_a_tavily_key(
    monkeypatch: pytest.MonkeyPatch, fake_response: ChatResponse
) -> None:
    monkeypatch.setattr("eval.invokers.settings.tavily_api_key", "fake-key")
    monkeypatch.setattr(
        "eval.invokers.run_rag_with_trace", lambda question, flags: (fake_response, [])
    )
    resp, _ = ServiceInvoker().invoke("What's the weather?", {}, "web_fallback")
    assert resp is fake_response
