"""The RAG orchestrator: retrieve -> spotlight -> generate, with cache read/write.

`run_rag_with_trace` is the primary test seam named in spec.md's Testing
Decisions: it returns the answer *and* the chunks retrieved to produce it, so
retrieval quality is testable without the cache, HTTP, or (later) the graph
in the way. The eval harness (ticket #33) calls it directly.

`run_rag` is what `/query` actually calls: a cache read, and on a miss, a
call to the trace function followed by a cache write.

`_retrieve` branches on `flags["search_mode"]` (dense / sparse / hybrid), then
optionally reranks — HyDE/CRAG/self-reflection are later tickets, each adding
a real behaviour behind its own flag. `flags` is threaded through and cached
against in full regardless, because story #42 requires the rag_answer cache
key to include every flag: toggling one, even one this ticket doesn't act on
yet, must not return another profile's stale cached answer.

When `enable_rerank` is set, retrieval asks for `settings.reranker_initial_top_k`
candidates (a wider pool than the final `top_k`) so the cross-encoder has a real
shortlist to re-sort — the gold chunk hybrid search buried at rank 8 only has a
chance to reach the top 5 if it was actually retrieved in the first place.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.models import ChatResponse, ResponseMetadata, RetrievedChunk, RetrievedChunkPreview
from app.security.spotlighting import spotlight_chunks
from app.security.system_prompt import SYSTEM_PROMPT
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate_text
from app.services.query_cache_service import query_cache
from app.services.reranker_service import rerank
from app.services.vector_store import hybrid_search, search, sparse_search

_CHUNK_PREVIEW_CHARS = 200


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Question: {question}\n\n"
        f"{spotlight_chunks(chunks)}\n\n"
        "Answer the question using only the retrieved context above, and cite "
        "the source filename(s) you used."
    )


def _retrieval_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return max(0.0, min(1.0, chunks[0].score))


def _sources(chunks: list[RetrievedChunk]) -> list[str]:
    # dict.fromkeys dedupes while keeping first-seen order (same trick used in
    # embedding_service.py for deduping a batch of texts).
    return list(dict.fromkeys(chunk.source for chunk in chunks))


def _retrieve(question: str, query_vector: list[float], flags: dict[str, Any]) -> list[RetrievedChunk]:
    top_k = int(flags.get("top_k", 5))
    search_mode = flags.get("search_mode", "dense")
    enable_rerank = flags.get("enable_rerank", False)

    # top_k can be requested up to 50 (QueryRequest); reranker_initial_top_k
    # (20) is only a *floor* on the candidate pool, not a cap on top_k itself.
    retrieve_k = max(top_k, settings.reranker_initial_top_k) if enable_rerank else top_k

    if search_mode == "sparse":
        chunks = sparse_search(question, top_k=retrieve_k)
    elif search_mode == "hybrid":
        chunks = hybrid_search(question, query_vector, top_k=retrieve_k)
    else:
        chunks = search(query_vector, top_k=retrieve_k)

    if enable_rerank:
        chunks = rerank(question, chunks)

    return chunks[:top_k]


def run_rag_with_trace(
    question: str, flags: dict[str, Any]
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    """Retrieve, generate, and return the response *and* the chunks used.

    No cache read or write — this is the uncached seam tests and the eval
    harness call directly.
    """
    query_vector = embed_texts([question])[0]
    chunks = _retrieve(question, query_vector, flags)

    llm_response = generate_text(_build_prompt(question, chunks), system_prompt=SYSTEM_PROMPT)

    response = ChatResponse(
        answer=llm_response.text,
        sources=_sources(chunks),
        retrieval_score=_retrieval_score(chunks),
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
