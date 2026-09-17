"""The settings object loads, has the documented defaults, and honours env overrides.

Default-value assertions run against a fresh `Settings(_env_file=None)` so a developer's
local `.env` (which `.env.example` tells them to create) can't break the suite.
"""

import pytest

from app.config import Settings, settings


@pytest.fixture
def defaults() -> Settings:
    return Settings(_env_file=None)


def test_settings_singleton_importable() -> None:
    # This is the ticket's acceptance check: `from app.config import settings` works.
    assert isinstance(settings, Settings)


def test_model_defaults(defaults: Settings) -> None:
    assert defaults.llm_model_answer == "gpt-5.4-mini"
    assert defaults.llm_model_grader == "gpt-5.5"
    assert defaults.embedding_model == "text-embedding-3-small"


def test_retrieval_defaults_match_spec(defaults: Settings) -> None:
    assert defaults.rrf_k == 60
    assert defaults.reranker_initial_top_k == 20
    assert defaults.crag_relevance_threshold == 0.7
    assert defaults.reflection_min_score == 0.85
    assert defaults.max_reflection_retries == 2
    assert defaults.hyde_num_hypotheses == 3


def test_cache_ttls_match_spec(defaults: Settings) -> None:
    assert defaults.cache_ttl_embeddings == 604_800  # 7 days
    assert defaults.cache_ttl_rag == 3_600  # 1 hour
    assert defaults.cache_ttl_sql_gen == 86_400  # 24 hours
    assert defaults.cache_ttl_sql_result == 900  # 15 minutes
    assert defaults.cache_ttl_intent == 86_400  # 24 hours


def test_security_and_budget_defaults(defaults: Settings) -> None:
    assert defaults.rate_limit_requests == 20
    assert defaults.rate_limit_window_seconds == 60
    assert defaults.max_tokens_per_user_daily == 100_000
    assert defaults.jwt_algorithm == "HS256"


def test_technique_toggle_defaults_agree_with_query_schema(defaults: Settings) -> None:
    # The QueryRequest schema is the authoritative per-request default; these
    # advisory flags must not contradict it.
    from app.models import QueryRequest

    req = QueryRequest(question="what is a pod?")
    assert defaults.hyde_enabled_by_default is req.enable_hyde is False
    assert defaults.reranking_enabled_by_default is req.enable_rerank is False
    assert defaults.crag_enabled_by_default is req.enable_crag is True
    assert defaults.self_reflective_enabled_by_default is req.enable_self_reflective is False
    assert defaults.adaptive_retrieval_enabled_by_default is req.enable_adaptive_retrieval is False


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RRF_K", "42")
    monkeypatch.setenv("RERANKER_BACKEND", "voyage")
    fresh = Settings(_env_file=None)
    assert fresh.rrf_k == 42
    assert fresh.reranker_backend == "voyage"
