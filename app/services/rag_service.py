"""The RAG orchestrator: retrieve -> spotlight -> generate, with cache read/write.

`run_rag_with_trace` is the primary test seam named in spec.md's Testing
Decisions: it returns the answer *and* the chunks retrieved to produce it, so
retrieval quality is testable without the cache, HTTP, or (later) the graph
in the way. The eval harness (ticket #33) calls it directly.

`run_rag` is what `/query` actually calls: a cache read, and on a miss, a
call to the trace function followed by a cache write.

Only dense search runs today — hybrid/rerank/HyDE/CRAG/self-reflection are
later tickets, each adding a real behaviour behind its own flag. `flags` is
threaded through and cached against in full regardless, because story #42
requires the rag_answer cache key to include every flag: toggling one, even
one this ticket doesn't act on yet, must not return another profile's stale
cached answer.
"""

from __future__ import annotations

from typing import Any

from app.models import ChatResponse, ResponseMetadata, RetrievedChunk, RetrievedChunkPreview
from app.security.spotlighting import spotlight_chunks
from app.security.system_prompt import SYSTEM_PROMPT
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate_text
from app.services.query_cache_service import query_cache
from app.services.vector_store import search

_CHUNK_PREVIEW_CHARS = 200


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Question: {question}\n\n"
        f"{spotlight_chunks(chunks)}\n\n"
        "Answer the question using only the retrieved context above, and cite "
        "the source filename(s) you used."
    )


def _confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return max(0.0, min(1.0, chunks[0].score))


def _sources(chunks: list[RetrievedChunk]) -> list[str]:
    # dict.fromkeys dedupes while keeping first-seen order (same trick used in
    # embedding_service.py for deduping a batch of texts).
    return list(dict.fromkeys(chunk.source for chunk in chunks))


def run_rag_with_trace(
    question: str, flags: dict[str, Any]
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    """Retrieve, generate, and return the response *and* the chunks used.

    No cache read or write — this is the uncached seam tests and the eval
    harness call directly.
    """
    top_k = int(flags.get("top_k", 5))
    query_vector = embed_texts([question])[0]
    chunks = search(query_vector, top_k=top_k)

    llm_response = generate_text(_build_prompt(question, chunks), system_prompt=SYSTEM_PROMPT)

    response = ChatResponse(
        answer=llm_response.text,
        sources=_sources(chunks),
        confidence=_confidence(chunks),
        cache_hit=False,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=[
                RetrievedChunkPreview(
                    text=chunk.text[:_CHUNK_PREVIEW_CHARS], source=chunk.source, score=chunk.score
                )
                for chunk in chunks
            ],
            cache_hit=False,
        ),
    )
    return response, chunks


def run_rag(question: str, flags: dict[str, Any]) -> ChatResponse:
    """The cached entry point `/query` calls: cache read, else run + write."""
    cached = query_cache.get_rag_answer(question, flags)
    if cached is not None:
        response = ChatResponse.model_validate(cached)
        response.cache_hit = True
        response.metadata.cache_hit = True
        return response

    response, _chunks = run_rag_with_trace(question, flags)
    query_cache.set_rag_answer(question, response.model_dump(mode="json"), flags)
    return response
