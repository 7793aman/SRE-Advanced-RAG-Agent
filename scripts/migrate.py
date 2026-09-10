"""Apply every SQL file in ``seed/migrations/`` in filename order.

Idempotent — re-running lands the same schema. ``001`` uses
``CREATE TABLE IF NOT EXISTS`` and keeps its data; ``003`` is a data seed that
DROPs and rebuilds its own tables each run (never touches ``users``).
``seed_db.py`` runs these migrations plus the demo users, and later the corpus.

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
