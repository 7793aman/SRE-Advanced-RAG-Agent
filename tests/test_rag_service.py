"""Unit seam: the RAG service's traced entry point (spec.md's primary test
seam). Embedding, vector search, the LLM, and the cache are all faked, so
these tests check *orchestration* (retrieve -> spotlight -> generate ->
cache) without touching OpenAI, Qdrant, or Redis.
"""

from __future__ import annotations

import pytest

from app.models import RetrievedChunk
from app.services.llm_service import LLMResponse
from app.services.query_cache_service import MemoryBackend, QueryCacheService

_CHUNKS = [
    RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=0.9),
    RetrievedChunk(text="A Deployment manages replica Pods.", source="deployments.html", score=0.7),
]
_FLAGS = {
    "search_mode": "dense",
    "enable_rerank": False,
    "enable_hyde": False,
    "enable_crag": True,
    "enable_self_reflective": False,
    "top_k": 5,
}


class _FakeGenerate:
    """Records every prompt it's asked to answer; returns a fixed reply."""

    def __init__(self, text: str = "A Pod is the smallest deployable unit. [pods.html]") -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return LLMResponse(text=self.text, prompt_tokens=50, completion_tokens=10, total_tokens=60)


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> QueryCacheService:
    cache = QueryCacheService(backend=MemoryBackend())
    monkeypatch.setattr("app.services.rag_service.query_cache", cache)
    return cache


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    calls: list[dict] = []

    def _search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return list(_CHUNKS)

    monkeypatch.setattr("app.services.rag_service.search", _search)
    return calls


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.setattr(
        "app.services.rag_service.embed_texts", lambda texts: [[0.1, 0.2] for _ in texts]
    )


@pytest.fixture
def fake_generate(monkeypatch: pytest.MonkeyPatch) -> _FakeGenerate:
    fake = _FakeGenerate()
    monkeypatch.setattr("app.services.rag_service.generate_text", fake)
    return fake


# --- run_rag_with_trace: the uncached, traced entry point ------------------


def test_returns_the_llm_answer_and_the_chunks_it_retrieved(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    response, chunks = run_rag_with_trace("What is a Pod?", _FLAGS)

    assert response.answer == fake_generate.text
    assert chunks == _CHUNKS


def test_passes_top_k_from_flags_to_search(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    run_rag_with_trace("What is a Pod?", {**_FLAGS, "top_k": 3})

    assert fake_search[0]["top_k"] == 3


def test_sources_are_deduplicated_in_first_seen_order(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    dup_chunks = [_CHUNKS[0], _CHUNKS[1], _CHUNKS[0]]
    import app.services.rag_service as rag_service

    rag_service.search = lambda *a, **k: dup_chunks  # type: ignore[assignment]

    response, _ = run_rag_with_trace("What is a Pod?", _FLAGS)

    assert response.sources == ["pods.html", "deployments.html"]


def test_retrieval_score_is_the_top_chunks_score(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    response, _ = run_rag_with_trace("What is a Pod?", _FLAGS)

    assert response.retrieval_score == pytest.approx(0.9)


def test_no_retrieved_chunks_gives_zero_retrieval_score_and_no_sources(
    fresh_cache, fake_embed, fake_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.rag_service import run_rag_with_trace

    monkeypatch.setattr("app.services.rag_service.search", lambda *a, **k: [])

    response, chunks = run_rag_with_trace("What is a Pod?", _FLAGS)

    assert response.retrieval_score == 0.0
    assert response.sources == []
    assert chunks == []


def test_prompt_sent_to_the_llm_includes_the_question_and_spotlighted_chunks(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    run_rag_with_trace("What is a Pod?", _FLAGS)

    prompt = fake_generate.calls[0]["prompt"]
    assert "What is a Pod?" in prompt
    assert "pods.html" in prompt
    assert "A Pod is the smallest deployable unit." in prompt


def test_uses_the_hardened_system_prompt(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.security.system_prompt import SYSTEM_PROMPT
    from app.services.rag_service import run_rag_with_trace

    run_rag_with_trace("What is a Pod?", _FLAGS)

    assert fake_generate.calls[0]["system_prompt"] == SYSTEM_PROMPT


def test_response_metadata_carries_a_preview_of_each_retrieved_chunk(
    fresh_cache, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    response, _ = run_rag_with_trace("What is a Pod?", _FLAGS)

    assert [p.source for p in response.metadata.retrieved_chunks] == [
        "pods.html",
        "deployments.html",
    ]
    assert response.metadata.route == "rag"


def test_trace_never_reads_or_writes_the_cache(
    fresh_cache: QueryCacheService, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag_with_trace

    run_rag_with_trace("What is a Pod?", _FLAGS)

    assert fresh_cache.stats()["rag_answer"] == {"hits": 0, "misses": 0, "sets": 0, "hit_rate": 0.0}


# --- run_rag: the cached entry point ----------------------------------------


def test_cache_miss_calls_the_llm_and_stores_the_result(
    fresh_cache: QueryCacheService, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag

    response = run_rag("What is a Pod?", _FLAGS)

    assert response.answer == fake_generate.text
    assert response.cache_hit is False
    assert len(fake_generate.calls) == 1
    assert fresh_cache.get_rag_answer("What is a Pod?", _FLAGS) is not None


def test_repeat_call_is_a_cache_hit_and_does_not_call_the_llm_again(
    fresh_cache: QueryCacheService, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag

    run_rag("What is a Pod?", _FLAGS)
    response = run_rag("What is a Pod?", _FLAGS)

    assert response.cache_hit is True
    assert response.metadata.cache_hit is True
    assert len(fake_generate.calls) == 1  # only the first call hit the LLM


def test_different_flags_are_a_cache_miss_even_for_the_same_question(
    fresh_cache: QueryCacheService, fake_search, fake_embed, fake_generate
) -> None:
    from app.services.rag_service import run_rag

    run_rag("What is a Pod?", _FLAGS)
    run_rag("What is a Pod?", {**_FLAGS, "enable_rerank": True})

    assert len(fake_generate.calls) == 2  # story #42: flags are part of the cache key


# --- enable_rerank: retrieve wide, rerank, then cut to top_k ----------------


def test_rerank_disabled_retrieves_only_top_k_and_never_calls_rerank(
    fresh_cache, fake_search, fake_embed, fake_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.rag_service import run_rag_with_trace

    monkeypatch.setattr(
        "app.services.rag_service.rerank",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerank should not be called")),
    )

    run_rag_with_trace("What is a Pod?", _FLAGS)

    assert fake_search[0]["top_k"] == 5  # _FLAGS["top_k"]


def test_rerank_enabled_retrieves_the_wider_candidate_pool(
    fresh_cache, fake_search, fake_embed, fake_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings
    from app.services.rag_service import run_rag_with_trace

    monkeypatch.setattr("app.services.rag_service.rerank", lambda question, chunks: chunks)

    run_rag_with_trace("What is a Pod?", {**_FLAGS, "enable_rerank": True})

    assert fake_search[0]["top_k"] == settings.reranker_initial_top_k


def test_rerank_enabled_with_top_k_above_the_initial_pool_still_retrieves_top_k(
    fresh_cache, fake_search, fake_embed, fake_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QueryRequest allows top_k up to 50 (app/models.py), well above the
    default reranker_initial_top_k of 20 — the wider pool is a floor, not a
    cap, so a caller asking for more chunks than that must still get them."""
    from app.services.rag_service import run_rag_with_trace

    monkeypatch.setattr("app.services.rag_service.rerank", lambda question, chunks: chunks)

    run_rag_with_trace("What is a Pod?", {**_FLAGS, "enable_rerank": True, "top_k": 30})

    assert fake_search[0]["top_k"] == 30


def test_rerank_enabled_reorders_before_cutting_to_top_k(
    fresh_cache, fake_embed, fake_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.rag_service import run_rag_with_trace

    wide_chunks = [
        RetrievedChunk(text="noise", source=f"doc{i}.html", score=0.1) for i in range(20)
    ]
    gold = RetrievedChunk(text="the real answer", source="gold.html", score=0.05)
    wide_chunks[7] = gold  # the gold chunk sits at rank 8, outside a top-5 cut

    monkeypatch.setattr("app.services.rag_service.search", lambda *a, **k: wide_chunks)
    monkeypatch.setattr(
        "app.services.rag_service.rerank",
        lambda question, chunks: [gold, *[c for c in chunks if c is not gold]],
    )

    _, chunks = run_rag_with_trace("What is a Pod?", {**_FLAGS, "enable_rerank": True, "top_k": 5})

    assert chunks[0] == gold
    assert len(chunks) == 5
