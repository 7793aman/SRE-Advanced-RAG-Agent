"""HyDE (hypothetical document embeddings): retrieve using LLM-generated
hypothetical answers instead of the raw question, to bridge the short-query
/ long-doc vocabulary gap (spec.md user story 13).

`hyde_search` generates `settings.hyde_num_hypotheses` hypothetical answers
to the question, keeps the original question as one more candidate, embeds
all of them (through the existing cached embedding service), runs one dense
vector-store search per embedding, and merges the results — deduping by
normalised chunk text and keeping whichever copy scored highest.

Hypothesis generation failure (LLM error/timeout) degrades to a plain dense
search on the original question only, never a failed request — the same
graceful-degradation contract as reranker_service's backend-failure
fallback (story #18).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from app.config import settings
from app.models import RetrievedChunk
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate_text
from app.services.vector_store import search

_HYPOTHESIS_PROMPT = (
    "Write a short passage (2-4 sentences) that could plausibly appear in "
    "technical documentation answering this question. State it as fact, "
    "even if you're not certain — do not hedge, and do not mention that "
    "this is hypothetical.\n\nQuestion: {question}"
)


def _generate_hypotheses(question: str, n: int) -> list[str]:
    if n <= 0:
        return []
    prompt = _HYPOTHESIS_PROMPT.format(question=question)
    # The N hypothesis generations don't depend on each other, so they run
    # concurrently — sequential would multiply this call's tail latency by
    # hyde_num_hypotheses for no benefit.
    with ThreadPoolExecutor(max_workers=n) as pool:
        responses = list(pool.map(lambda _: generate_text(prompt), range(n)))
    return [response.text.strip() for response in responses if response.text.strip()]


def _normalise(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _dedupe_keep_best(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = _normalise(chunk.text)
        current = best.get(key)
        if current is None or chunk.score > current.score:
            best[key] = chunk
    return list(best.values())


def hyde_search(
    question: str, top_k: int = 5, num_hypotheses: int | None = None
) -> list[RetrievedChunk]:
    """Search with N LLM-generated hypothetical answers plus the original
    question, merged and deduped keeping the best-scored copy of each
    chunk, sorted most relevant first and cut to `top_k`."""
    n = num_hypotheses if num_hypotheses is not None else settings.hyde_num_hypotheses

    try:
        hypotheses = _generate_hypotheses(question, n)
    except Exception:  # noqa: BLE001 — degrade to dense search on the question, never fail the request
        logger.warning(
            "HyDE hypothesis generation failed; falling back to dense search on the question"
        )
        hypotheses = []

    # The question is always embedded and searched too, even on the happy
    # path: it's cheap insurance for cases where the raw query already
    # embeds close enough, and it's what's left after a generation failure.
    texts = [*hypotheses, question]
    vectors = embed_texts(texts)

    # One vector-store search per embedding, run concurrently for the same
    # reason as the hypothesis generations above: each search is independent
    # of the others.
    with ThreadPoolExecutor(max_workers=len(vectors)) as pool:
        result_lists = list(pool.map(lambda vector: search(vector, top_k=top_k), vectors))
    all_chunks = [chunk for chunks in result_lists for chunk in chunks]

    deduped = _dedupe_keep_best(all_chunks)
    return sorted(deduped, key=lambda chunk: chunk.score, reverse=True)[:top_k]
