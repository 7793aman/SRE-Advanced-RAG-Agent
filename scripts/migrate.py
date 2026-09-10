"""Apply every SQL file in ``seed/migrations/`` in filename order.

Idempotent — each migration uses ``CREATE TABLE IF NOT EXISTS`` and friends, so
re-running is safe. Ticket #21's ``seed_db.py`` builds the demo users and the
operational data on top of this.

    make migrate      # or: uv run python scripts/migrate.py
"""

from __future__ import annotations

from pathlib import Path

from app.db import run_sql_file

_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "seed" / "migrations"


def main() -> None:
    paths = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not paths:
        print(f"no migrations found in {_MIGRATIONS_DIR}")
        return
    for path in paths:
        print(f"applying {path.name}")
        run_sql_file(path)
    print(f"done — {len(paths)} migration(s) applied")


if __name__ == "__main__":
    main()
