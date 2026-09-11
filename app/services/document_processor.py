"""Parse + chunk a document (PDF/DOCX/HTML/TXT) into {text, source, page_number?}.

Uses docling to read the file and split it into chunks. Text is cut first
along structural boundaries (headings, paragraphs, table rows) and only
further split by size if a piece is still too large — that's what docling's
`HybridChunker` means by "hybrid."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _page_number(chunk: Any) -> int | None:
    """Best-effort page number: only PDFs carry one, and not every chunk does."""
    doc_items = getattr(getattr(chunk, "meta", None), "doc_items", None)
    if not doc_items:
        return None
    prov = getattr(doc_items[0], "prov", None)
    if not prov:
        return None
    return int(prov[0].page_no)


class DocumentProcessor:
    def __init__(self) -> None:
        self._converter = _build_converter()
        self._chunker = HybridChunker()

    def process_document(self, file_path: str) -> list[dict[str, Any]]:
        document = self._converter.convert(file_path).document
        source = Path(file_path).name

        chunks: list[dict[str, Any]] = []
        for chunk in self._chunker.chunk(document):
            entry: dict[str, Any] = {"text": chunk.text, "source": source}
            page_number = _page_number(chunk)
            if page_number is not None:
                entry["page_number"] = page_number
            chunks.append(entry)

        logger.info("processed {} chunks from {}", len(chunks), file_path)
        return chunks
