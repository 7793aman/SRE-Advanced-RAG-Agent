"""CRAG (Corrective Retrieval-Augmented Generation): grade retrieved chunks
for relevance before generation, and correct weak retrieval with a web
search rather than let a confident wrong answer through (spec.md user
stories 14-15; "CRAG service").

`evaluate_and_correct` is the entry point `rag_service._retrieve` calls after
reranking. It grades the final chunk set with an LLM-as-judge call (a
cross-encoder rerank only orders candidates against each other; it can't say
"none of these are good enough" — only an absolute score against a fixed
threshold can), then acts on `config.py`'s two thresholds:

  score >= crag_relevance_threshold   -> correct: use the chunks as-is
  crag_ambiguous_threshold <= score   -> ambiguous: keep the corpus chunks
    < crag_relevance_threshold           *and* add web results
  score < crag_ambiguous_threshold    -> incorrect: discard the corpus,
                                          answer from web results only

An empty `chunks` list has nothing to grade, so it skips the LLM call
entirely and goes straight to the web fallback (an automatic "incorrect").

Tavily is the one place in this flow that fails loudly (see
`web_search_service`'s `WebSearchUnconfiguredError`) — everything here
catches that, logs it, and degrades to the original corpus chunks
unchanged, so a missing `TAVILY_API_KEY` or a Tavily outage never fails
the request. A grader-call outage degrades the same way — a broken judge
isn't evidence the retrieval is bad, so it's treated as "skip correction",
not "assume incorrect" — matching the graceful-degradation contract
reranker_service and hyde_service already follow (story #18).
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import ValidationError

from app.config import settings
from app.models import CRAGEvaluation, RetrievedChunk
from app.services.llm_service import generate_json
from app.services.web_search_service import WebSearchUnconfiguredError, web_search

_GRADER_SYSTEM_PROMPT = (
    "You are a strict retrieval-relevance grader for a Kubernetes "
    "operations knowledge base. Given a question and a set of retrieved "
    "passages, judge whether the passages, taken together, actually "
    "contain the answer to the question — not just whether they're "
    "topically similar.\n\n"
    "Respond with a JSON object only, no other text:\n"
    '{"relevance_score": <float 0.0-1.0>, '
    '"relevance_label": "correct" | "ambiguous" | "incorrect", '
    '"confidence": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
)


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)


def _grade(question: str, chunks: list[RetrievedChunk]) -> CRAGEvaluation:
    if not chunks:
        return CRAGEvaluation(
            relevance_score=0.0,
            relevance_label="incorrect",
            confidence=1.0,
            reasoning="nothing was retrieved",
        )

    prompt = f"Question: {question}\n\nRetrieved passages:\n{_format_chunks(chunks)}"
    try:
        response = generate_json(prompt, system_prompt=_GRADER_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 — a grader outage degrades to "skip correction", never fails the request
        logger.warning("CRAG grader call failed; keeping retrieval as-is")
        return CRAGEvaluation(
            relevance_score=1.0,
            relevance_label="correct",
            confidence=0.0,
            reasoning="grader call failed",
        )

    try:
        return CRAGEvaluation.model_validate(json.loads(response.text))
    except (json.JSONDecodeError, ValidationError):
        logger.warning("CRAG grader returned malformed JSON; treating retrieval as incorrect")
        return CRAGEvaluation(
            relevance_score=0.0,
            relevance_label="incorrect",
            confidence=0.0,
            reasoning="grader output was malformed",
        )


def evaluate_and_correct(
    question: str, chunks: list[RetrievedChunk], top_k: int = 5
) -> list[RetrievedChunk]:
    """Grade `chunks` for relevance to `question`; below the relevance
    threshold, correct with a Tavily web search. Always logs the grade.

    `top_k` is the caller's requested chunk count (`rag_service._retrieve`'s
    own `top_k`) — it bounds how many web results are requested and how many
    chunks a correction returns, so a corrected answer doesn't silently fall
    back to Tavily's own default result count regardless of what was asked
    for."""
    evaluation = _grade(question, chunks)
    logger.info(
        "CRAG grade {:.2f} ({}) for question={!r}: {}",
        evaluation.relevance_score,
        evaluation.relevance_label or "ungraded",
        question,
        evaluation.reasoning,
    )

    if evaluation.relevance_score >= settings.crag_relevance_threshold:
        return chunks

    try:
        web_chunks = web_search(question, max_results=top_k)
    except WebSearchUnconfiguredError:
        logger.warning(
            "CRAG wanted a web fallback but Tavily is unconfigured; keeping corpus chunks as-is"
        )
        return chunks
    except Exception:  # noqa: BLE001 — a Tavily outage degrades to the corpus, never fails the request
        logger.warning("Tavily web search failed; keeping corpus chunks as-is")
        return chunks

    if evaluation.relevance_score >= settings.crag_ambiguous_threshold:
        combined = chunks + web_chunks
        return combined[:top_k]

    return web_chunks[:top_k]
