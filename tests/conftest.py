"""Shared fixtures.

The auth HTTP tests need a live local Postgres (per spec's Testing Decisions).
If one isn't reachable they skip rather than fail, so the pure-unit seams
(rate-limiter math, token mint/verify) still run anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app
from app.middleware.rate_limiter import MemoryBackend, rate_limiter
from app.security import content_guard
from app.security.token_budget import MemoryBudgetBackend, token_budget

_MIGRATION = Path(__file__).resolve().parents[1] / "seed" / "migrations" / "001_create_users.sql"


@pytest.fixture(scope="session", autouse=True)
def _limiter_uses_memory() -> None:
    """Pin the shared limiter to an in-process backend for the whole run.

    Without this, a checkout whose `.env` configures Upstash would make the auth
    HTTP tests hit (and pollute) a live Redis. Tests must not depend on it.
    """
    rate_limiter._backend = MemoryBackend()


@pytest.fixture(autouse=True)
def _strong_jwt_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give tokens a real signing key regardless of the local `.env`."""
    if len(settings.jwt_secret) < 32:
        monkeypatch.setattr(settings, "jwt_secret", "test-secret-key-at-least-32-bytes-long!!")


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    rate_limiter.reset()


@pytest.fixture(scope="session", autouse=True)
def _budget_uses_memory() -> None:
    """Same reason as the limiter: never let a test touch a live Redis."""
    token_budget._backend = MemoryBudgetBackend()


@pytest.fixture(autouse=True)
def _reset_token_budget() -> None:
    token_budget.reset()


@pytest.fixture(autouse=True)
def _guard_models_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The llm-guard models are slow and need the network. By default the guard runs
    its regex fallback; tests that care about the scanners patch the loaders."""
    monkeypatch.setattr(content_guard, "_load_input_scanners", lambda: None)
    monkeypatch.setattr(content_guard, "_load_output_scanners", lambda: None)


@pytest.fixture(scope="session")
def db_ready() -> None:
    try:
        db.run_sql_file(_MIGRATION)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")


@pytest.fixture(scope="session")
def migrations_applied() -> None:
    """All of `seed/migrations/` applied, or skip when Postgres is unreachable."""
    try:
        db.run_migrations()
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")


@pytest.fixture
def clean_users(db_ready: None) -> Iterator[None]:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")
    yield
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
