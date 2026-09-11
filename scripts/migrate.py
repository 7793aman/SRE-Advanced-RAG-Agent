"""Apply every SQL file in ``seed/migrations/`` in filename order.

Idempotent — each migration uses ``CREATE TABLE IF NOT EXISTS`` and, for the
data seed (``003``), ``INSERT ... ON CONFLICT DO NOTHING``, so re-running is
safe. Thin CLI over ``app.db.run_migrations``; ``seed_db.py`` calls the same
function before creating the demo users and the corpus.

    make migrate      # or: uv run python scripts/migrate.py
"""

from __future__ import annotations

from app.db import run_migrations


def main() -> None:
    applied = run_migrations()
    print(f"done — {len(applied)} migration(s) applied: {', '.join(applied)}")


if __name__ == "__main__":
    main()
