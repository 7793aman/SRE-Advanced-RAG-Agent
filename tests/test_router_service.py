"""Unit seam: classify_intent(), with the classifier LLM call and the cache
faked. Nothing here calls OpenAI or a real cache backend.
"""

from __future__ import annotations

import json

import pytest

from app.services.llm_service import LLMResponse
from app.services.query_cache_service import MemoryBackend, QueryCacheService


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> QueryCacheService:
    cache = QueryCacheService(backend=MemoryBackend())
    monkeypatch.setattr("app.services.router_service.query_cache", cache)
    return cache


def _router_response(intent: str) -> LLMResponse:
    return LLMResponse(text=json.dumps({"intent": intent, "reasoning": "test"}))


# --- cache behaviour ---------------------------------------------------------


def test_cache_hit_returns_the_cached_intent_without_calling_the_llm(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    fresh_cache.set_intent("how many P1 incidents last month?", "sql")

    def _boom(*a: object, **k: object) -> LLMResponse:
        raise AssertionError("should not call the LLM on a cache hit")

    monkeypatch.setattr(router_service, "generate_json", _boom)

    assert router_service.classify_intent("how many P1 incidents last month?") == "sql"


def test_cache_miss_classifies_and_caches_the_result(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(router_service, "generate_json", lambda *a, **k: _router_response("sql"))

    intent = router_service.classify_intent("how many P1 incidents last month?")

    assert intent == "sql"
    assert fresh_cache.get_intent("how many P1 incidents last month?") == "sql"


def test_second_identical_question_reuses_the_cache(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    calls = 0

    def _fake_generate_json(*a: object, **k: object) -> LLMResponse:
        nonlocal calls
        calls += 1
        return _router_response("rag")

    monkeypatch.setattr(router_service, "generate_json", _fake_generate_json)

    router_service.classify_intent("how does a Deployment roll out?")
    router_service.classify_intent("how does a Deployment roll out?")

    assert calls == 1


# --- classifications ----------------------------------------------------------


def test_documentation_question_classifies_as_rag(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(router_service, "generate_json", lambda *a, **k: _router_response("rag"))

    assert router_service.classify_intent("How does a Deployment roll out?") == "rag"


def test_operational_question_classifies_as_sql(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(router_service, "generate_json", lambda *a, **k: _router_response("sql"))

    assert router_service.classify_intent("how many P1 incidents last month") == "sql"


def test_combined_question_classifies_as_hybrid(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(
        router_service, "generate_json", lambda *a, **k: _router_response("hybrid")
    )

    assert router_service.classify_intent("list P1 incidents with remediation docs") == "hybrid"


# --- graceful degradation ------------------------------------------------------


def test_llm_call_failure_degrades_to_rag_without_caching(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    def _boom(*a: object, **k: object) -> LLMResponse:
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr(router_service, "generate_json", _boom)

    assert router_service.classify_intent("how many P1 incidents last month?") == "rag"
    assert fresh_cache.get_intent("how many P1 incidents last month?") is None


def test_malformed_json_degrades_to_rag_without_caching(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(
        router_service, "generate_json", lambda *a, **k: LLMResponse(text="not valid json")
    )

    assert router_service.classify_intent("how many P1 incidents last month?") == "rag"
    assert fresh_cache.get_intent("how many P1 incidents last month?") is None


def test_unexpected_intent_label_degrades_to_rag_without_caching(
    fresh_cache: QueryCacheService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import router_service

    monkeypatch.setattr(
        router_service, "generate_json", lambda *a, **k: _router_response("documents")
    )

    assert router_service.classify_intent("how many P1 incidents last month?") == "rag"
    assert fresh_cache.get_intent("how many P1 incidents last month?") is None
