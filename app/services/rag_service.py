"""The RAG orchestrator: retrieve -> spotlight -> generate, with cache read/write.

`run_rag_with_trace` is the primary test seam named in spec.md's Testing
Decisions: it returns the answer *and* the chunks retrieved to produce it, so
retrieval quality is testable without the cache, HTTP, or (later) the graph
in the way. The eval harness (ticket #33) calls it directly.

`run_rag` is what `/query` actually calls: a cache read, and on a miss, a
call to the trace function followed by a cache write.

`_retrieve` branches on `flags["search_mode"]` (dense / sparse / hybrid), then
optionally reranks and grades the result with CRAG. `flags` is threaded
through and cached against in full regardless, because story #42 requires
the rag_answer cache key to include every flag: toggling one, even one a
given retrieval path doesn't act on, must not return another profile's stale
cached answer.

When `enable_hyde` is set, it takes over the retrieval step entirely in
place of `search_mode`'s dense/sparse/hybrid branch: HyDE already does its
own dense search per hypothesis (see `hyde_service.py`), so there's nothing
left for `search_mode` to pick between.

When `enable_rerank` is set, retrieval asks for `settings.reranker_initial_top_k`
candidates (a wider pool than the final `top_k`) so the cross-encoder has a real
shortlist to re-sort — the gold chunk hybrid search buried at rank 8 only has a
chance to reach the top 5 if it was actually retrieved in the first place.

When `enable_crag` is set (the default), the final `top_k` chunks are graded
for relevance before generation; a weak grade corrects them with a Tavily web
search rather than generating confidently from noise — see `crag_service.py`.

Two Self-RAG flags exist and are independent of each other (see
`reflection_service.py`):

  - `enable_adaptive_retrieval` controls `needs_retrieval`: whether the
    question is general knowledge the model can answer directly. If so,
    retrieval and CRAG are skipped entirely, and generation runs under
    `GENERAL_KNOWLEDGE_SYSTEM_PROMPT` instead of the corpus-only
    `SYSTEM_PROMPT` (`route="rag_general_knowledge"` in the response
    metadata) — `SYSTEM_PROMPT` demands an "I don't know" for anything not in
    the retrieved context, which is exactly wrong when there deliberately is
    none. This flag works whether or not `enable_self_reflective` is set: a
    skipped-retrieval answer is just returned as-is if reflection is off.
  - `enable_self_reflective` controls the critique-and-retry loop: after
    each generation, `reflect` critiques the answer against the *original*
    question, and `should_regenerate` — bounded by
    `settings.max_reflection_retries` — decides whether to loop again with a
    sharpened question. The reflection loop only ever regenerates; it never
    re-decides retrieve-vs-skip, so a regeneration stays in whichever regime
    `needs_retrieval` chose at the top (or the corpus regime, if
    `enable_adaptive_retrieval` was never on to begin with): a corpus
    question reruns retrieval (and CRAG) on the refined question, since the
    sharpened question is only worth anything against context retrieved for
    it, while a general-knowledge question regenerates again with no
    retrieval. `reflection_iterations`, `reflection_score`, and
    `refined_question` surface this in the response metadata regardless of
    which path ran.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.models import ChatResponse, ResponseMetadata, RetrievedChunk, RetrievedChunkPreview
from app.security.spotlighting import spotlight_chunks
from app.security.system_prompt import GENERAL_KNOWLEDGE_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.services.crag_service import evaluate_and_correct
from app.services.embedding_service import embed_texts
from app.services.hyde_service import hyde_search
from app.services.llm_service import LLMResponse, generate_text
from app.services.query_cache_service import query_cache
from app.services.reflection_service import needs_retrieval, reflect, should_regenerate
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


def _retrieve(
    question: str, query_vector: list[float], flags: dict[str, Any]
) -> list[RetrievedChunk]:
    top_k = int(flags.get("top_k", 5))
    search_mode = flags.get("search_mode", "dense")
    enable_rerank = flags.get("enable_rerank", False)
    enable_hyde = flags.get("enable_hyde", False)
    enable_crag = flags.get("enable_crag", True)

    # top_k can be requested up to 50 (QueryRequest); reranker_initial_top_k
    # (20) is only a *floor* on the candidate pool, not a cap on top_k itself.
    retrieve_k = max(top_k, settings.reranker_initial_top_k) if enable_rerank else top_k

    if enable_hyde:
        chunks = hyde_search(question, top_k=retrieve_k)
    elif search_mode == "sparse":
        chunks = sparse_search(question, top_k=retrieve_k)
    elif search_mode == "hybrid":
        chunks = hybrid_search(question, query_vector, top_k=retrieve_k)
    else:
        chunks = search(query_vector, top_k=retrieve_k)

    if enable_rerank:
        chunks = rerank(question, chunks)

    chunks = chunks[:top_k]

    if enable_crag:
        chunks = evaluate_and_correct(question, chunks, top_k=top_k)

    return chunks


def _retrieve_and_generate(
    question: str, flags: dict[str, Any]
) -> tuple[list[RetrievedChunk], LLMResponse]:
    query_vector = embed_texts([question])[0]
    chunks = _retrieve(question, query_vector, flags)
    llm_response = generate_text(_build_prompt(question, chunks), system_prompt=SYSTEM_PROMPT)
    return chunks, llm_response


def run_rag_with_trace(
    question: str, flags: dict[str, Any]
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    """Retrieve, generate, and return the response *and* the chunks used.

    No cache read or write — this is the uncached seam tests and the eval
    harness call directly.
    """
    enable_self_reflective = flags.get("enable_self_reflective", False)
    enable_adaptive_retrieval = flags.get("enable_adaptive_retrieval", False)
    skip_retrieval = enable_adaptive_retrieval and not needs_retrieval(question)

    if skip_retrieval:
        route = "rag_general_knowledge"
        chunks: list[RetrievedChunk] = []
        llm_response = generate_text(question, system_prompt=GENERAL_KNOWLEDGE_SYSTEM_PROMPT)
    else:
        route = "rag"
        chunks, llm_response = _retrieve_and_generate(question, flags)

    reflection_iterations = 0
    reflection_score: float | None = None
    refined_question: str | None = None

    if enable_self_reflective:
        reflection = reflect(question, llm_response.text, chunks)
        reflection_score = reflection.reflection_score
        while should_regenerate(reflection, reflection_iterations):
            reflection_iterations += 1
            refined_question = reflection.refined_question or question
            # The reflection loop only regenerates — it never revisits the
            # retrieve/skip decision `needs_retrieval` already made, so a
            # general-knowledge answer that scores low is retried the same
            # way (no corpus lookup), not silently upgraded to a real search.
            if skip_retrieval:
                llm_response = generate_text(
                    refined_question, system_prompt=GENERAL_KNOWLEDGE_SYSTEM_PROMPT
                )
            else:
                chunks, llm_response = _retrieve_and_generate(refined_question, flags)
            reflection = reflect(question, llm_response.text, chunks)
            reflection_score = reflection.reflection_score

    response = ChatResponse(
        answer=llm_response.text,
        sources=_sources(chunks),
        retrieval_score=_retrieval_score(chunks),
        cache_hit=False,
        metadata=ResponseMetadata(
            route=route,
            retrieved_chunks=[
                RetrievedChunkPreview(
                    text=chunk.text[:_CHUNK_PREVIEW_CHARS], source=chunk.source, score=chunk.score
                )
                for chunk in chunks
            ],
            cache_hit=False,
            reflection_iterations=reflection_iterations,
            reflection_score=reflection_score,
            refined_question=refined_question,
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
