"""Shared prompt-formatting for retrieved chunks — every service that hands
chunks to an LLM as judge/grader context (crag_service's grader,
reflection_service's critic) formats them the same way."""

from __future__ import annotations

from app.models import RetrievedChunk


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)
