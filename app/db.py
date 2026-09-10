"""Thin psycopg2 helper for the operational Postgres database.

Shared infrastructure, not a test seam: auth (#20) reads/writes `users` here,
and later slices (ops-data seeding #21, the SQL service #30) reuse the same
`connection()` context manager and `run_sql_file()` migration runner.

`connection()` commits on clean exit, rolls back on exception, always closes.
Cursors yield `NamedTuple` rows so callers can say `row.username`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import psycopg2
import psycopg2.extensions
from psycopg2.extras import NamedTupleCursor

from app.config import settings


@contextlib.contextmanager
def connection() -> Iterator[psycopg2.extensions.connection]:
    """Open a connection, commit on success, roll back on error, always close."""
    conn = psycopg2.connect(settings.database_url, cursor_factory=NamedTupleCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_sql_file(path: str | Path) -> None:
    """Execute every statement in a `.sql` file in one transaction."""
    sql = Path(path).read_text(encoding="utf-8")
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
