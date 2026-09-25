"""Unit seam: QueryCacheService's public API, backed by MemoryBackend (no Redis).

Mirrors test_rate_limiter.py's pattern: inject the in-memory backend directly
so these tests are deterministic and never touch real Redis.
"""

from __future__ import annotations

import hashlib

import pytest

from app.services.query_cache_service import MemoryBackend, QueryCacheService

_MODEL = "text-embedding-3-small"


@pytest.fixture
def cache() -> QueryCacheService:
    return QueryCacheService(backend=MemoryBackend())


@pytest.fixture
def shared_backend() -> MemoryBackend:
    """A backend outliving any single `QueryCacheService` instance — stands
    in for Upstash surviving a process restart, unlike `cache`'s backend
    which is fresh per test."""
    return MemoryBackend()


# --- key derivation ----------------------------------------------------------


def test_embedding_key_is_deterministic_sha256(cache: QueryCacheService) -> None:
    expected = f"embedding:{hashlib.sha256(b'text-embedding-3-small:hello world').hexdigest()}"
    assert cache.embedding_key("hello world", "text-embedding-3-small") == expected
    assert cache.embedding_key("hello world", "text-embedding-3-small") == cache.embedding_key(
        "hello world", "text-embedding-3-small"
    )


def test_embedding_key_is_case_and_whitespace_sensitive(cache: QueryCacheService) -> None:
    model = "text-embedding-3-small"
    assert cache.embedding_key("Hello", model) != cache.embedding_key("hello", model)
    assert cache.embedding_key("hello ", model) != cache.embedding_key("hello", model)


def test_embedding_key_changes_with_the_model(cache: QueryCacheService) -> None:
    # Two models produce different (and often differently-sized) vectors for
    # the same text, so a cache hit must never cross model boundaries.
    small = cache.embedding_key("hello", "text-embedding-3-small")
    large = cache.embedding_key("hello", "text-embedding-3-large")
    assert small != large


def test_intent_key_normalises_case_and_surrounding_whitespace(cache: QueryCacheService) -> None:
    assert cache.intent_key("Why is my Pod crashing?") == cache.intent_key(
        "  why is my pod crashing?  "
    )


def test_different_tiers_hash_the_same_text_to_different_keys(cache: QueryCacheService) -> None:
    assert cache.embedding_key("x", "text-embedding-3-small") != cache.intent_key("x")


def test_rag_answer_key_changes_with_flags(cache: QueryCacheService) -> None:
    base = cache.rag_answer_key("what is a pod?")
    with_rerank = cache.rag_answer_key("what is a pod?", flags={"enable_rerank": True})
    with_hyde = cache.rag_answer_key("what is a pod?", flags={"enable_hyde": True})

    assert base != with_rerank
    assert base != with_hyde
    assert with_rerank != with_hyde


def test_rag_answer_key_is_stable_for_the_same_flags_regardless_of_order(
    cache: QueryCacheService,
) -> None:
    a = cache.rag_answer_key("q", flags={"enable_rerank": True, "enable_hyde": False})
    b = cache.rag_answer_key("q", flags={"enable_hyde": False, "enable_rerank": True})
    assert a == b


def test_sql_result_key_normalises_whitespace_and_case(cache: QueryCacheService) -> None:
    a = cache.sql_result_key("SELECT * FROM pods")
    b = cache.sql_result_key("  select   *   from   pods  ")
    assert a == b


# --- tier stats ----------------------------------------------------------------


def test_stats_start_at_zero_for_every_tier(cache: QueryCacheService) -> None:
    stats = cache.stats()
    assert set(stats) == {"embedding", "intent", "rag_answer", "sql_gen", "sql_result"}
    for tier_stats in stats.values():
        assert tier_stats == {"hits": 0, "misses": 0, "sets": 0, "hit_rate": 0.0}


def test_get_before_set_records_a_miss(cache: QueryCacheService) -> None:
    assert cache.get_embedding("unseen text", _MODEL) is None
    assert cache.stats()["embedding"] == {"hits": 0, "misses": 1, "sets": 0, "hit_rate": 0.0}


def test_set_then_get_records_a_set_and_a_hit(cache: QueryCacheService) -> None:
    cache.set_embedding("hello", [0.1, 0.2], _MODEL)
    assert cache.get_embedding("hello", _MODEL) == [0.1, 0.2]

    stats = cache.stats()["embedding"]
    assert stats["sets"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["hit_rate"] == pytest.approx(1.0)


def test_hit_rate_reflects_a_mix_of_hits_and_misses(cache: QueryCacheService) -> None:
    cache.set_intent("what is a pod?", "rag")
    cache.get_intent("what is a pod?")  # hit
    cache.get_intent("what is a pod?")  # hit
    cache.get_intent("unrelated question")  # miss

    stats = cache.stats()["intent"]
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3)


def test_tiers_do_not_share_stats(cache: QueryCacheService) -> None:
    cache.set_embedding("x", [1.0], _MODEL)
    cache.get_embedding("x", _MODEL)

    intent_stats = cache.stats()["intent"]
    assert intent_stats == {"hits": 0, "misses": 0, "sets": 0, "hit_rate": 0.0}


# --- clear() ---------------------------------------------------------------------


def test_clear_evicts_every_tier_and_resets_stats(cache: QueryCacheService) -> None:
    cache.set_embedding("a", [1.0], _MODEL)
    cache.set_intent("q", "rag")
    cache.set_rag_answer("q", {"answer": "42"})
    cache.set_sql_generation("q", "SELECT 1")
    cache.set_sql_result("SELECT 1", [{"a": 1}])

    cache.clear()

    for tier_stats in cache.stats().values():
        assert tier_stats == {"hits": 0, "misses": 0, "sets": 0, "hit_rate": 0.0}

    assert cache.get_embedding("a", _MODEL) is None
    assert cache.get_intent("q") is None


def test_clear_evicts_entries_written_by_an_earlier_process(shared_backend: MemoryBackend) -> None:
    """Regression: `clear()` used to delete only keys its OWN in-process
    bookkeeping remembered writing — against the real (Upstash, persistent
    across restarts) backend, that missed every key a previous process
    instance had written, so `clear()` reported success but left stale
    entries in place indefinitely. A fresh `QueryCacheService` (simulating a
    server restart) must still be able to clear entries an earlier instance
    wrote to the same backend, which means `clear()` has to scan the backend
    itself rather than trust any in-process record of past writes."""
    earlier_process = QueryCacheService(backend=shared_backend)
    earlier_process.set_sql_result("SELECT 1", [{"a": 1}])

    restarted_process = QueryCacheService(backend=shared_backend)
    assert restarted_process.get_sql_result("SELECT 1") is not None  # sanity: it's really there

    restarted_process.clear()

    assert restarted_process.get_sql_result("SELECT 1") is None


# --- ping() ------------------------------------------------------------------


def test_ping_is_true_for_a_working_backend(cache: QueryCacheService) -> None:
    assert cache.ping() is True


def test_ping_is_false_when_the_backend_raises() -> None:
    class _BrokenBackend:
        def get(self, key: str) -> str | None:
            raise ConnectionError("Redis is down")

        def set(self, key: str, value: str, ttl_seconds: int) -> None:
            raise ConnectionError("Redis is down")

        def delete(self, keys: list[str]) -> None:
            raise ConnectionError("Redis is down")

    cache = QueryCacheService(backend=_BrokenBackend())

    assert cache.ping() is False


def test_ping_does_not_affect_any_tiers_stats(cache: QueryCacheService) -> None:
    cache.ping()

    for tier_stats in cache.stats().values():
        assert tier_stats == {"hits": 0, "misses": 0, "sets": 0, "hit_rate": 0.0}
    assert cache.get_rag_answer("q") is None
    assert cache.get_sql_generation("q") is None
    assert cache.get_sql_result("SELECT 1") is None
