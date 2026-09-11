"""HTTP seam: `/admin/health` (public) and `/admin/cache/*` (admin-only).

The live dependency checks (Postgres, Qdrant) are faked so these tests don't
need real infrastructure — they check this endpoint's own logic (how it
combines checks into a status, and who is allowed to call it), not whether
Postgres itself is actually reachable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.middleware.auth import create_access_token


@pytest.fixture
def admin_token() -> str:
    return create_access_token(user_id=1, username="admin@demo.local", is_admin=True)


@pytest.fixture
def user_token() -> str:
    return create_access_token(user_id=2, username="agent@demo.local", is_admin=False)


# --- GET /admin/health -------------------------------------------------------


def test_health_needs_no_authentication(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.admin._check_postgres", lambda: True)
    monkeypatch.setattr("app.api.admin._check_vector_store", lambda: True)
    monkeypatch.setattr("app.api.admin.query_cache.ping", lambda: True)
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test")

    resp = client.get("/admin/health")

    assert resp.status_code == 200


def test_health_is_healthy_when_every_dependency_is_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.admin._check_postgres", lambda: True)
    monkeypatch.setattr("app.api.admin._check_vector_store", lambda: True)
    monkeypatch.setattr("app.api.admin.query_cache.ping", lambda: True)
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test")

    body = client.get("/admin/health").json()

    assert body["postgres"] is True
    assert body["vector_store"] is True
    assert body["cache"] is True
    assert body["llm_api"] is True
    assert body["status"] == "healthy"


def test_health_is_degraded_when_postgres_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.admin._check_postgres", lambda: False)
    monkeypatch.setattr("app.api.admin._check_vector_store", lambda: True)
    monkeypatch.setattr("app.api.admin.query_cache.ping", lambda: True)
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test")

    body = client.get("/admin/health").json()

    assert body["postgres"] is False
    assert body["status"] == "degraded"


def test_health_is_degraded_when_no_openai_key_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.admin._check_postgres", lambda: True)
    monkeypatch.setattr("app.api.admin._check_vector_store", lambda: True)
    monkeypatch.setattr("app.api.admin.query_cache.ping", lambda: True)
    monkeypatch.setattr("app.config.settings.openai_api_key", "")

    body = client.get("/admin/health").json()

    assert body["llm_api"] is False
    assert body["status"] == "degraded"


def test_health_is_degraded_when_the_cache_is_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.admin._check_postgres", lambda: True)
    monkeypatch.setattr("app.api.admin._check_vector_store", lambda: True)
    monkeypatch.setattr("app.api.admin.query_cache.ping", lambda: False)
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test")

    body = client.get("/admin/health").json()

    assert body["cache"] is False
    assert body["status"] == "degraded"


# --- GET /admin/cache/stats ---------------------------------------------------


def test_cache_stats_requires_a_token(client: TestClient) -> None:
    resp = client.get("/admin/cache/stats")

    assert resp.status_code == 401


def test_cache_stats_rejects_a_non_admin_user(client: TestClient, user_token: str) -> None:
    resp = client.get("/admin/cache/stats", headers={"Authorization": f"Bearer {user_token}"})

    assert resp.status_code == 403


def test_cache_stats_returns_the_query_caches_stats(
    client: TestClient, admin_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_stats = {"embedding": {"hits": 3, "misses": 1, "sets": 1, "hit_rate": 0.75}}
    monkeypatch.setattr("app.api.admin.query_cache.stats", lambda: fixed_stats)

    resp = client.get("/admin/cache/stats", headers={"Authorization": f"Bearer {admin_token}"})

    assert resp.status_code == 200
    assert resp.json() == fixed_stats


# --- POST /admin/cache/clear ---------------------------------------------------


def test_cache_clear_requires_a_token(client: TestClient) -> None:
    resp = client.post("/admin/cache/clear")

    assert resp.status_code == 401


def test_cache_clear_rejects_a_non_admin_user(client: TestClient, user_token: str) -> None:
    resp = client.post("/admin/cache/clear", headers={"Authorization": f"Bearer {user_token}"})

    assert resp.status_code == 403


def test_cache_clear_calls_the_query_caches_clear(
    client: TestClient, admin_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr("app.api.admin.query_cache.clear", lambda: calls.append(True))

    resp = client.post("/admin/cache/clear", headers={"Authorization": f"Bearer {admin_token}"})

    assert resp.status_code == 200
    assert calls == [True]
