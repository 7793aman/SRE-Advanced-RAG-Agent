"""`scripts/seed_db.py` — the seed orchestrator.

The `--no-ingest` path (migrations + demo users) end to end against live
Postgres; skips if Postgres is unreachable.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg2
import pytest

from app import db
from app.middleware.auth import verify_password
from scripts import seed_db


@pytest.fixture
def migrated_db() -> Iterator[None]:
    try:
        seed_db.run_migrations()
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")
    yield
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")


def _users() -> dict[str, tuple[str, bool]]:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT username, password_hash, is_admin FROM users")
        return {r.username: (r.password_hash, r.is_admin) for r in cur.fetchall()}


def test_run_migrations_reports_the_files_applied() -> None:
    try:
        applied = seed_db.run_migrations()
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    assert "001_create_users.sql" in applied
    assert "003_seed_k8s_ops.sql" in applied


def test_seed_users_creates_both_demo_users(migrated_db: None) -> None:
    seed_db.seed_users()
    users = _users()
    assert set(users) == {"agent@demo.local", "admin@demo.local"}
    assert users["admin@demo.local"][1] is True
    assert users["agent@demo.local"][1] is False


def test_seed_users_passwords_verify(migrated_db: None) -> None:
    seed_db.seed_users()
    users = _users()
    assert verify_password("agent123", users["agent@demo.local"][0])
    assert verify_password("admin123", users["admin@demo.local"][0])


def test_seed_users_is_idempotent(migrated_db: None) -> None:
    seed_db.seed_users()
    seed_db.seed_users()
    assert len(_users()) == 2


def test_seed_users_refreshes_an_existing_row(migrated_db: None) -> None:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s)",
            ("admin@demo.local", "stale-hash", False),
        )
    seed_db.seed_users()
    hash_, is_admin = _users()["admin@demo.local"]
    assert is_admin is True
    assert verify_password("admin123", hash_)


def test_main_no_ingest_returns_zero_and_seeds(migrated_db: None) -> None:
    assert seed_db.main(["--no-ingest"]) == 0
    assert set(_users()) == {"agent@demo.local", "admin@demo.local"}


def test_parse_noise_sample_rejects_garbage() -> None:
    with pytest.raises(SystemExit):
        seed_db._parse_noise_sample("lots")


def test_parse_noise_sample_accepts_all_and_ints() -> None:
    assert seed_db._parse_noise_sample("all") == "all"
    assert seed_db._parse_noise_sample("300") == 300
