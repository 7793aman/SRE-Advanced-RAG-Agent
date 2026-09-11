"""Unit seam: DocumentProcessor, run against a real small signal file.

Uses the real docling library (no fakes) — PDFs are slow and can download OCR
models on first use, but plain text is fast and needs no network, so this
stays a real, fast, offline-friendly test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.services.document_processor import DocumentProcessor, _page_number

_SIGNAL_TXT = (
    Path(__file__).resolve().parents[1]
    / "seed"
    / "docs"
    / "true_data"
    / "concepts__security__overview.txt"
)


def test_process_document_returns_chunks_with_text_and_source() -> None:
    processor = DocumentProcessor()

    chunks = processor.process_document(str(_SIGNAL_TXT))

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["text"].strip() != ""
        assert chunk["source"] == "concepts__security__overview.txt"


def test_process_document_uses_the_filename_not_the_full_path_as_source() -> None:
    processor = DocumentProcessor()

    chunks = processor.process_document(str(_SIGNAL_TXT))

    assert "/" not in chunks[0]["source"]
    assert chunks[0]["source"] == _SIGNAL_TXT.name


def test_process_document_omits_page_number_for_a_text_file() -> None:
    processor = DocumentProcessor()

    chunks = processor.process_document(str(_SIGNAL_TXT))

    assert all("page_number" not in chunk for chunk in chunks)


# --- _page_number(): a plain function, tested directly with a crafted input,
# no fakes/injection needed for this since it's not part of DocumentProcessor's
# constructor. docling nests the page number three attributes deep on a real
# chunk (chunk.meta.doc_items[0].prov[0].page_no); the .txt-based tests above
# never exercise that path since text files have no pages at all.


def test_page_number_extracts_the_page_from_docling_style_metadata() -> None:
    doc_item = SimpleNamespace(prov=[SimpleNamespace(page_no=5)])
    chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[doc_item]))

    assert _page_number(chunk) == 5


def test_page_number_is_none_when_a_chunk_has_no_metadata_at_all() -> None:
    assert _page_number(SimpleNamespace()) is None
