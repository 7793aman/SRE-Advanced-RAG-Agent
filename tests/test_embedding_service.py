"""Unit seam: embed_texts()'s cache read-through, with OpenAI and the cache faked.

Nothing here touches a real OpenAI call or a real Redis — the fake client
records what it was asked to embed, and a fresh in-memory cache is swapped in
per test so tests can't see each other's state.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.query_cache_service import MemoryBackend, QueryCacheService


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddingsAPI:
    """Records every batch it's asked to embed; returns a fixed vector per text."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text
        self.calls: list[list[str]] = []

    def create(self, input: list[str], model: str) -> _FakeEmbeddingResponse:  # noqa: A002
        self.calls.append(list(input))
        return _FakeEmbeddingResponse([_FakeEmbeddingItem(self._vectors_by_text[t]) for t in input])


class _FakeOpenAIClient:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.embeddings = _FakeEmbeddingsAPI(vectors_by_text)


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> QueryCacheService:
    cache = QueryCacheService(backend=MemoryBackend())
    monkeypatch.setattr("app.services.embedding_service.query_cache", cache)
    return cache


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIClient:
    client = _FakeOpenAIClient({"pod": [0.1, 0.2], "deployment": [0.3, 0.4]})
    monkeypatch.setattr("app.services.embedding_service._get_client", lambda: client)
    return client


def test_embed_texts_returns_empty_list_for_no_input(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    assert embed_texts([]) == []
    assert fake_client.embeddings.calls == []


def test_embed_texts_calls_openai_once_for_a_cache_miss_and_caches_the_result(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    vectors = embed_texts(["pod"])

    assert vectors == [[0.1, 0.2]]
    assert fake_client.embeddings.calls == [["pod"]]
    assert fresh_cache.get_embedding("pod", settings.embedding_model) == [0.1, 0.2]


def test_repeat_call_is_a_cache_hit_with_no_further_openai_call(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    embed_texts(["pod"])
    vectors = embed_texts(["pod"])

    assert vectors == [[0.1, 0.2]]
    assert fake_client.embeddings.calls == [["pod"]]  # only the first call hit OpenAI


def test_mixed_batch_sends_only_misses_and_preserves_order(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    embed_texts(["pod"])  # warm the cache for "pod"
    vectors = embed_texts(["pod", "deployment"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_client.embeddings.calls == [["pod"], ["deployment"]]


def test_duplicate_text_in_one_call_is_sent_to_openai_only_once(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    vectors = embed_texts(["pod", "pod"])

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert fake_client.embeddings.calls == [["pod"]]  # not [["pod", "pod"]]


def test_stats_show_embedding_tier_hits_and_misses(
    fresh_cache: QueryCacheService, fake_client: _FakeOpenAIClient
) -> None:
    from app.services.embedding_service import embed_texts

    embed_texts(["pod"])  # miss + set
    embed_texts(["pod"])  # hit

    stats = fresh_cache.stats()["embedding"]
    assert stats == {"hits": 1, "misses": 1, "sets": 1, "hit_rate": pytest.approx(0.5)}
