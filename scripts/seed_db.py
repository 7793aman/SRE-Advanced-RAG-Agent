"""Seed orchestrator: schema migrations, the two demo users, and (optionally) the
document corpus ingestion.

    uv run python scripts/seed_db.py                     # everything
    uv run python scripts/seed_db.py --no-ingest         # DB only, no vector store
    uv run python scripts/seed_db.py --noise-sample 500  # 47 signal + 500 noise
    uv run python scripts/seed_db.py --noise-sample all  # 47 signal + all noise

The signal corpus (`seed/docs/true_data/`, 47 files) is always ingested in full.
The noise corpus (`seed/docs/noisy_data/`) is sampled to a configurable size with
a fixed seed, so the same `--noise-sample N` always picks the same N files.

Corpus wiring: the noise corpus is staged directly at `seed/docs/noisy_data/`
(gitignored bodies, see `seed/docs/README.md`). No symlink — the seeder reads
that path as-is.

Doc ingestion depends on the embedding + vector-store services (ticket #22); until
those land, run with `--no-ingest`.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from app.db import connection, run_sql_file
from app.middleware.auth import hash_password

_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = _ROOT / "seed" / "migrations"
DOCS_DIR = _ROOT / "seed" / "docs"

SIGNAL_SUBDIR = "true_data"
NOISE_SUBDIR = "noisy_data"

# Parsers the document processor (ticket #22) can handle.
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}

# Fixed seed for noise sampling — stable selection across runs and machines.
NOISE_SAMPLE_SEED = 42

# username, plaintext password, is_admin
DEMO_USERS: tuple[tuple[str, str, bool], ...] = (
    ("agent@demo.local", "agent123", False),
    ("admin@demo.local", "admin123", True),
)


@dataclass(frozen=True)
class CorpusSelection:
    """The files a seed run will ingest: all signal, a sample of noise."""

    signal: list[Path]
    noise: list[Path]

    @property
    def total(self) -> int:
        return len(self.signal) + len(self.noise)


def _list_docs(directory: Path) -> list[Path]:
    """Every supported document under ``directory``, sorted for a stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name != ".gitkeep"
    )


def select_corpus(noise_sample: int | Literal["all"], docs_dir: Path = DOCS_DIR) -> CorpusSelection:
    """Pick the corpus for a seed run.

    Signal is always taken in full. Noise is sampled to ``noise_sample`` files
    with a fixed seed; ``"all"`` (or a count at/above the pool size) takes every
    noise file, ``0`` takes none.
    """
    signal = _list_docs(docs_dir / SIGNAL_SUBDIR)
    noise_pool = _list_docs(docs_dir / NOISE_SUBDIR)

    if noise_sample == "all":
        noise = noise_pool
    else:
        n = int(noise_sample)
        if n < 0:
            raise ValueError(f"noise_sample must be >= 0 or 'all', got {noise_sample!r}")
        if n >= len(noise_pool):
            noise = noise_pool
        else:
            noise = sorted(random.Random(NOISE_SAMPLE_SEED).sample(noise_pool, n))

    return CorpusSelection(signal=signal, noise=noise)


def run_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every `.sql` file in filename order. Returns the names applied."""
    paths = sorted(migrations_dir.glob("*.sql"))
    if not paths:
        logger.warning("no migrations found in {}", migrations_dir)
        return []
    for path in paths:
        logger.info("applying migration {}", path.name)
        run_sql_file(path)
    return [p.name for p in paths]


def seed_users(users: Iterable[tuple[str, str, bool]] = DEMO_USERS) -> None:
    """Upsert the demo users. Idempotent — re-running refreshes their hashes."""
    with connection() as conn, conn.cursor() as cur:
        for username, password, is_admin in users:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, is_admin)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    is_admin = EXCLUDED.is_admin
                """,
                (username, hash_password(password), is_admin),
            )
            logger.info("seeded user {} (admin={})", username, is_admin)


def ingest_corpus(selection: CorpusSelection) -> None:
    """Parse, embed, and upsert every selected file into the vector store.

    Imports the ingestion services lazily so `--no-ingest` runs (and this
    module's import) don't need ticket #22's dependencies.
    """
    from app.services.document_processor import DocumentProcessor
    from app.services.embedding_service import embed_texts
    from app.services.vector_store import upsert_chunks

    from app.models import RetrievedChunk

    processor = DocumentProcessor()
    ordered = [(p, "signal") for p in selection.signal]
    ordered += [(p, "noise") for p in selection.noise]
    logger.info(
        "ingesting {} files ({} signal + {} noise)",
        selection.total,
        len(selection.signal),
        len(selection.noise),
    )
    ingested = failed = chunk_count = 0
    for idx, (path, label) in enumerate(ordered, start=1):
        try:
            meta = processor.process_document(str(path))
            if not meta:
                logger.warning("[{}/{}] {} {} → 0 chunks", idx, selection.total, label, path.name)
                failed += 1
                continue
            chunks = [RetrievedChunk(text=m["text"], source=m["source"]) for m in meta]
            upsert_chunks(chunks, embed_texts([c.text for c in chunks]))
            ingested += 1
            chunk_count += len(chunks)
        except Exception:  # noqa: BLE001 — one bad file must not abort the seed
            logger.exception("[{}/{}] failed {} {}", idx, selection.total, label, path.name)
            failed += 1
    logger.info("ingestion done — {} files, {} chunks, {} failed", ingested, chunk_count, failed)


def _parse_noise_sample(raw: str) -> int | Literal["all"]:
    if raw == "all":
        return "all"
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"--noise-sample must be an integer or 'all', got {raw!r}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the operational DB and document corpus")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="run migrations + demo users only; skip vector-store ingestion",
    )
    parser.add_argument(
        "--noise-sample",
        default="150",
        metavar="N|all",
        help="how many noise files to ingest (fixed seed); default 150",
    )
    args = parser.parse_args(argv)
    noise_sample = _parse_noise_sample(args.noise_sample)

    logger.info("running migrations")
    applied = run_migrations()
    logger.info("applied {} migration(s): {}", len(applied), ", ".join(applied))

    logger.info("seeding demo users")
    seed_users()

    if args.no_ingest:
        logger.info("--no-ingest set; skipping document ingestion")
        return 0

    selection = select_corpus(noise_sample)
    if selection.total == 0:
        logger.warning("no documents found under {} — nothing to ingest", DOCS_DIR)
        return 0
    ingest_corpus(selection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
