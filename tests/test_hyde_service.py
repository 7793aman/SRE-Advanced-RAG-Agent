"""Unit seam: hyde_search(), with hypothesis generation, embedding, and
vector search all faked — same shape as test_reranker_service.py. Nothing
here calls OpenAI or Qdrant.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.models import RetrievedChunk
from app.services.llm_service import LLMResponse


@pytest.fixture(autouse=True)
def _num_hypotheses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "hyde_num_hypotheses", 2)


def _fake_embed(texts: list[str]) -> list[list[float]]:
    # One distinguishable "vector" per input text, so fake search can key off it.
    return [[float(len(t))] for t in texts]


# --- hypothesis generation + embedding + per-hypothesis search -------------


def test_generates_configured_number_of_hypotheses_plus_the_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    calls: list[str] = []

    def _fake_generate(prompt: str, **kwargs: object) -> LLMResponse:
        calls.append(prompt)
        return LLMResponse(text=f"hypothesis for: {prompt}")

    monkeypatch.setattr(hyde_service, "generate_text", _fake_generate)
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)
    monkeypatch.setattr(hyde_service, "search", lambda vector, top_k=5: [])

    hyde_service.hyde_search("why is my pod OOMKilled?", top_k=5)

    assert len(calls) == 2  # settings.hyde_num_hypotheses


def test_hypothesis_generation_uses_a_system_prompt_and_a_non_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical, low-temperature completions for the same question would
    waste every extra hypothesis — they all embed to roughly the same
    vector, so a higher temperature is what actually gives HyDE N distinct
    angles to search with."""
    from app.services import hyde_service

    calls: list[dict] = []

    def _fake_generate(prompt: str, **kwargs: object) -> LLMResponse:
        calls.append({"prompt": prompt, **kwargs})
        return LLMResponse(text="a hypothesis")

    monkeypatch.setattr(hyde_service, "generate_text", _fake_generate)
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)
    monkeypatch.setattr(hyde_service, "search", lambda vector, top_k=5: [])

    hyde_service.hyde_search("why is my pod OOMKilled?")

    assert calls[0]["prompt"] == "why is my pod OOMKilled?"
    assert calls[0]["system_prompt"]
    assert calls[0]["temperature"] > 0


def test_searches_once_per_hypothesis_plus_once_for_the_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    monkeypatch.setattr(
        hyde_service, "generate_text", lambda prompt, **kw: LLMResponse(text="a hypothesis")
    )
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)

    search_calls: list[list[float]] = []

    def _fake_search(vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        search_calls.append(vector)
        return []

    monkeypatch.setattr(hyde_service, "search", _fake_search)

    hyde_service.hyde_search("why is my pod OOMKilled?", top_k=5)

    assert len(search_calls) == 3  # 2 hypotheses + the original question


def test_passes_top_k_through_to_each_per_hypothesis_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    monkeypatch.setattr(
        hyde_service, "generate_text", lambda prompt, **kw: LLMResponse(text="a hypothesis")
    )
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)

    top_ks: list[int] = []
    monkeypatch.setattr(
        hyde_service,
        "search",
        lambda vector, top_k=5: top_ks.append(top_k) or [],  # type: ignore[func-returns-value]
    )

    hyde_service.hyde_search("why is my pod OOMKilled?", top_k=7)

    assert top_ks == [7, 7, 7]


# --- dedupe: keep the highest-scored duplicate ------------------------------


def test_dedupe_keeps_the_higher_scored_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import hyde_service

    weak = RetrievedChunk(
        text="OOMKilled means the kernel killed the container.", source="oom.html", score=0.3
    )
    strong = RetrievedChunk(
        text="OOMKilled means the kernel killed the container.", source="oom.html", score=0.9
    )

    result = hyde_service._dedupe_keep_best([weak, strong])

    assert result == [strong]


def test_dedupe_ignores_whitespace_and_case_when_matching_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    weak = RetrievedChunk(
        text="  OOMKilled Means   the kernel killed it.  ", source="oom.html", score=0.2
    )
    strong = RetrievedChunk(
        text="OOMKilled means the kernel killed it.", source="oom.html", score=0.8
    )

    result = hyde_service._dedupe_keep_best([weak, strong])

    assert result == [strong]


def test_dedupe_keeps_distinct_chunks_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import hyde_service

    a = RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=0.5)
    b = RetrievedChunk(
        text="A Deployment manages replica Pods.", source="deployments.html", score=0.4
    )

    result = hyde_service._dedupe_keep_best([a, b])

    assert {c.source for c in result} == {"pods.html", "deployments.html"}


def test_full_search_dedupes_across_hypotheses_and_sorts_by_score_descending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    monkeypatch.setattr(
        hyde_service, "generate_text", lambda prompt, **kw: LLMResponse(text="a hypothesis")
    )
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)

    gold_weak = RetrievedChunk(text="the real answer", source="gold.html", score=0.2)
    gold_strong = RetrievedChunk(text="the real answer", source="gold.html", score=0.95)
    noise = RetrievedChunk(text="unrelated noise", source="noise.html", score=0.5)

    # Per-vector searches run concurrently (see hyde_search), so which of the
    # 3 fixed result lists lands on which call is nondeterministic — but the
    # final merge+dedupe+sort is order-independent, so the assertions below
    # hold regardless of assignment order.
    import itertools
    import threading

    lock = threading.Lock()
    counter = itertools.count()
    result_lists = [[gold_weak, noise], [gold_strong], [noise]]

    def _fake_search(vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        with lock:
            i = next(counter)
        return result_lists[i]

    monkeypatch.setattr(hyde_service, "search", _fake_search)

    result = hyde_service.hyde_search("why is my pod OOMKilled?", top_k=5)

    assert result[0] == gold_strong
    assert len(result) == 2  # noise deduped to a single copy


def test_result_is_cut_to_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import hyde_service

    monkeypatch.setattr(
        hyde_service, "generate_text", lambda prompt, **kw: LLMResponse(text="a hypothesis")
    )
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)

    chunks = [
        RetrievedChunk(text=f"chunk {i}", source=f"doc{i}.html", score=i / 10) for i in range(10)
    ]
    monkeypatch.setattr(hyde_service, "search", lambda vector, top_k=5: chunks)

    result = hyde_service.hyde_search("why is my pod OOMKilled?", top_k=3)

    assert len(result) == 3
    assert [c.score for c in result] == [0.9, 0.8, 0.7]


# --- graceful degradation: LLM failure falls back to dense search ----------


def test_hypothesis_generation_failure_falls_back_to_dense_search_on_the_question_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import hyde_service

    def _boom(prompt: str, **kwargs: object) -> LLMResponse:
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(hyde_service, "generate_text", _boom)
    monkeypatch.setattr(hyde_service, "embed_texts", _fake_embed)

    search_calls: list[list[float]] = []

    def _fake_search(vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        search_calls.append(vector)
        return [RetrievedChunk(text="fallback result", source="fallback.html", score=0.5)]

    monkeypatch.setattr(hyde_service, "search", _fake_search)

    result = hyde_service.hyde_search("why is my pod OOMKilled?", top_k=5)

    assert len(search_calls) == 1  # only the original question, no hypotheses
    assert [c.source for c in result] == ["fallback.html"]
