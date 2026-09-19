"""Integration seam: `ingest_corpus` skips files the vector store already holds.

Real local Qdrant (throwaway collection); only the parser and the embedder are
faked, since those are slow and cost money.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.models import RetrievedChunk
from scripts.seed_db import CorpusSelection, ingest_corpus
from tests.test_vector_store import _unit_vector, qdrant_collection, qdrant_ready  # noqa: F401


def test_ingest_skips_files_already_in_the_vector_store(
    qdrant_collection: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import document_processor, embedding_service
    from app.services.vector_store import source_exists, upsert_chunks

    old, new = tmp_path / "old.md", tmp_path / "new.md"
    old.write_text("old")
    new.write_text("new")
    upsert_chunks([RetrievedChunk(text="old text", source="old.md")], [_unit_vector(0)])

    parsed: list[str] = []
    embedded: list[str] = []

    class FakeProcessor:
        def process_document(self, file_path: str) -> list[dict[str, Any]]:
            parsed.append(Path(file_path).name)
            return [{"text": f"text of {Path(file_path).name}", "source": Path(file_path).name}]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        embedded.extend(texts)
        return [_unit_vector(1) for _ in texts]

    monkeypatch.setattr(document_processor, "DocumentProcessor", FakeProcessor)
    monkeypatch.setattr(embedding_service, "embed_texts", fake_embed)

    ingest_corpus(CorpusSelection(signal=[old, new], noise=[]))

    assert parsed == ["new.md"]
    assert embedded == ["text of new.md"]
    assert source_exists("new.md") is True
