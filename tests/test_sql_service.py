"""Unit seam: the SQL service — the SELECT-only guard, cached SQL generation,
and cached execution with row serialisation. The LLM and the database
connection are faked; nothing here calls OpenAI or Postgres.

The last test group runs schema introspection against a live Postgres and
skips if one isn't reachable.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from decimal import Decimal

import psycopg2
import pytest

from app import db
from app.services import sql_service
from app.services.llm_service import LLMResponse
from app.services.query_cache_service import QueryCacheService


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> QueryCacheService:
    from app.services.query_cache_service import MemoryBackend

    cache = QueryCacheService(backend=MemoryBackend())
    monkeypatch.setattr(sql_service, "query_cache", cache)
    return cache


# --- SELECT-only guard -----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM incidents",
        "select cluster_id, count(*) from incidents group by cluster_id;",
        "  SELECT 1  ",
        "WITH p1 AS (SELECT * FROM incidents WHERE severity = 'P1') SELECT count(*) FROM p1",
        "SELECT * FROM alerts WHERE name = 'delete-me'",  # keyword inside a literal is fine
        "SELECT created_at, updated_by FROM pods",  # `created_at` / `updated_by` are not keywords
    ],
)
def test_guard_accepts_plain_selects(sql: str) -> None:
    assert sql_service.validate_select_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE incidents",
        "drop table incidents;",
        "DELETE FROM incidents",
        "UPDATE incidents SET severity = 'P4'",
        "INSERT INTO clusters VALUES (1)",
        "TRUNCATE incidents",
        "ALTER TABLE incidents ADD COLUMN x int",
        "CREATE TABLE t (id int)",
        "GRANT ALL ON incidents TO public",
        "COPY incidents TO '/tmp/x'",
        "SELECT * INTO backup FROM incidents",
        "SELECT 1; DROP TABLE incidents",
        "SELECT 1; SELECT 2",
        "SELECT 1 /* harmless */; DELETE FROM incidents",
        "WITH d AS (DELETE FROM incidents RETURNING *) SELECT * FROM d",
        "SELECT pg_sleep(10)",
        "SELECT * FROM users",
        "SELECT password_hash FROM public.users",
        "SELECT * FROM checkpoints",
        'SELECT * FROM "users"',
        "SELECT $$; DROP TABLE x; SELECT $$",
        "SELECT E'\\'' ; DROP TABLE x",
        "",
        "   ",
    ],
)
def test_guard_refuses_anything_that_is_not_a_plain_select(sql: str) -> None:
    with pytest.raises(sql_service.UnsafeSQLError):
        sql_service.validate_select_only(sql)


def test_guard_returns_the_sql_without_a_trailing_semicolon() -> None:
    assert sql_service.validate_select_only("SELECT 1;  ") == "SELECT 1"


# --- generation ------------------------------------------------------------------


def _fake_llm(monkeypatch: pytest.MonkeyPatch, payload: dict | str) -> list[dict]:
    calls: list[dict] = []
    text = payload if isinstance(payload, str) else json.dumps(payload)

    def _generate_json(prompt: str, system_prompt: str | None = None, **_: object) -> LLMResponse:
        calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return LLMResponse(text=text)

    monkeypatch.setattr(sql_service, "generate_json", _generate_json)
    monkeypatch.setattr(sql_service, "get_schema", lambda: "TABLE incidents (severity text)")
    return calls


def test_generate_sql_returns_the_llms_select_and_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_llm(monkeypatch, {"sql": "SELECT 1;", "explanation": "counts things"})

    generated = sql_service.generate_sql("how many things?")

    assert generated.sql == "SELECT 1"
    assert generated.explanation == "counts things"


def test_generate_sql_puts_the_live_schema_in_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_llm(monkeypatch, {"sql": "SELECT 1", "explanation": ""})

    sql_service.generate_sql("how many things?")

    assert "TABLE incidents (severity text)" in calls[0]["system_prompt"]
    assert calls[0]["prompt"] == "how many things?"


def test_generate_sql_is_cached_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_llm(monkeypatch, {"sql": "SELECT 1", "explanation": "e"})

    first = sql_service.generate_sql("how many things?")
    second = sql_service.generate_sql("  How many things?  ")

    assert len(calls) == 1
    assert second == first


def test_generate_sql_refuses_a_dangerous_statement_and_does_not_cache_it(
    monkeypatch: pytest.MonkeyPatch, _fresh_cache: QueryCacheService
) -> None:
    _fake_llm(monkeypatch, {"sql": "DROP TABLE incidents", "explanation": "oops"})

    with pytest.raises(sql_service.UnsafeSQLError):
        sql_service.generate_sql("drop everything")

    assert _fresh_cache.get_sql_generation("drop everything") is None


def test_generate_sql_raises_on_malformed_llm_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_llm(monkeypatch, "not json at all")

    with pytest.raises(sql_service.SQLGenerationError):
        sql_service.generate_sql("how many things?")


def test_generate_sql_raises_when_the_llm_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_: object, **__: object) -> LLMResponse:
        raise RuntimeError("openai down")

    monkeypatch.setattr(sql_service, "generate_json", _boom)
    monkeypatch.setattr(sql_service, "get_schema", lambda: "")

    with pytest.raises(sql_service.SQLGenerationError):
        sql_service.generate_sql("how many things?")


# --- execution -------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[dict], executed: list[str]) -> None:
        self._rows = rows
        self._executed = executed

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str) -> None:
        self._executed.append(sql)

    def fetchmany(self, size: int) -> list[dict]:
        return self._rows[:size]


class _FakeConn:
    def __init__(self, rows: list[dict], executed: list[str]) -> None:
        self._rows = rows
        self._executed = executed

    def cursor(self, **_: object) -> _FakeCursor:
        return _FakeCursor(self._rows, self._executed)


def _fake_db(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> list[str]:
    executed: list[str] = []

    @contextlib.contextmanager
    def _connection():  # noqa: ANN202
        yield _FakeConn(rows, executed)

    monkeypatch.setattr(sql_service, "connection", _connection)
    return executed


def test_execute_sql_returns_serialised_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_db(
        monkeypatch,
        [{"n": Decimal("2.50"), "at": datetime(2026, 1, 2, 3, 4, tzinfo=UTC), "name": "prod"}],
    )

    rows, cache_hit = sql_service.execute_sql("SELECT 1")

    assert rows == [{"n": 2.5, "at": "2026-01-02T03:04:00+00:00", "name": "prod"}]
    assert cache_hit is False
    json.dumps(rows)  # must be JSON-safe: it is checkpointed and cached


def test_execute_sql_refuses_a_non_select_without_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed = _fake_db(monkeypatch, [])

    with pytest.raises(sql_service.UnsafeSQLError):
        sql_service.execute_sql("DROP TABLE incidents")

    assert executed == []


def test_execute_sql_runs_read_only_with_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    executed = _fake_db(monkeypatch, [])

    sql_service.execute_sql("SELECT 1")

    assert any("READ ONLY" in stmt.upper() for stmt in executed)
    assert any("statement_timeout" in stmt for stmt in executed)
    assert executed[-1] == "SELECT 1"


def test_execute_sql_caps_the_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_db(monkeypatch, [{"i": i} for i in range(sql_service.MAX_ROWS + 50)])

    rows, _ = sql_service.execute_sql("SELECT i FROM t")
    assert len(rows) == sql_service.MAX_ROWS


def test_execute_sql_wraps_database_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextlib.contextmanager
    def _connection():  # noqa: ANN202
        raise psycopg2.OperationalError("connection refused")
        yield

    monkeypatch.setattr(sql_service, "connection", _connection)

    with pytest.raises(sql_service.SQLExecutionError):
        sql_service.execute_sql("SELECT 1")


def test_execute_sql_results_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    executed = _fake_db(monkeypatch, [{"n": 1}])

    first_rows, first_cache_hit = sql_service.execute_sql("SELECT 1")
    executed.clear()
    second_rows, second_cache_hit = sql_service.execute_sql("select   1")

    assert second_rows == first_rows
    assert executed == []
    # The cache-hit flag itself (issue #34: this used to never surface up to
    # `ChatResponse.cache_hit`, so the UI's "Cache: hit/miss" line was wrong
    # for every SQL-path answer) — first call is a fresh execution, the
    # normalised-identical second call must report the cache hit honestly.
    assert first_cache_hit is False
    assert second_cache_hit is True


# --- schema introspection (live Postgres) ------------------------------------------


@pytest.fixture
def live_db() -> None:
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.incidents')")
            if cur.fetchone()[0] is None:
                pytest.skip("ops schema not migrated (run `make migrate`)")
    except Exception:  # noqa: BLE001
        pytest.skip("Postgres unreachable")


def test_get_schema_lists_the_operational_tables_and_columns(live_db: None) -> None:
    schema = sql_service.get_schema()

    for table in ("clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"):
        assert f"TABLE {table}" in schema
    assert "severity" in schema
    assert "cluster_id" in schema


def test_get_schema_hides_the_users_table(live_db: None) -> None:
    assert "TABLE users" not in sql_service.get_schema()
    assert "password" not in sql_service.get_schema()


def test_a_real_select_runs_against_the_seeded_database(live_db: None) -> None:
    rows, _ = sql_service.execute_sql(
        "SELECT cluster_id, count(*) AS n FROM incidents WHERE severity = 'P1' "
        "GROUP BY cluster_id ORDER BY n DESC LIMIT 1"
    )

    assert len(rows) == 1
    assert set(rows[0]) == {"cluster_id", "n"}
