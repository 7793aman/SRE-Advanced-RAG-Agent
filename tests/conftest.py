"""Shared fixtures.

The auth HTTP tests need a live local Postgres (per spec's Testing Decisions).
If one isn't reachable they skip rather than fail, so the pure-unit seams
(rate-limiter math, token mint/verify) still run anywhere.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _point_tests_at_a_separate_database() -> None:
    """`clean_users` truncates `users` on every run — against the same
    database the live app/Streamlit demo uses, that wipes out whoever is
    logged in there mid-session. `app.config.settings` is a module-level
    singleton read once on import, so `DATABASE_URL` has to be overridden
    *before* anything below here imports `app.config` (directly or via
    `app.db`/`app.main`), landing every connection this test session makes
    on a `_test`-suffixed sibling database on the same Postgres server
    instead of the real one.

    Creates that sibling database on first use (e.g. a fresh worktree or
    CI box that's never run these tests against this Postgres before) so
    the isolation is automatic rather than a manual setup step someone has
    to remember. If Postgres isn't reachable at all, this is a no-op —
    `db_ready`/`migrations_applied` already skip cleanly in that case."""
    base_url = dotenv_values(_PROJECT_ROOT / ".env").get("DATABASE_URL") or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/adv_rag"
    )
    parts = urlsplit(base_url)
    test_db_name = f"{parts.path.lstrip('/')}_test"
    os.environ["DATABASE_URL"] = urlunsplit(parts._replace(path=f"/{test_db_name}"))

    maintenance_url = urlunsplit(parts._replace(path="/postgres"))
    try:
        conn = psycopg2.connect(maintenance_url)
    except psycopg2.OperationalError:
        return
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db_name,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{test_db_name}"')
    finally:
        conn.close()


_point_tests_at_a_separate_database()

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.middleware.rate_limiter import MemoryBackend, rate_limiter  # noqa: E402
from app.security import content_guard  # noqa: E402
from app.security.token_budget import MemoryBudgetBackend, token_budget  # noqa: E402

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
