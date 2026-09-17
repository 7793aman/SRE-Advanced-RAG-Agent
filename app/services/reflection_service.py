"""Self-RAG: critique the generated answer and regenerate with a sharpened
question if it's weak, bounded by a retry limit (spec.md user story 16;
"Self-reflective service"). Also covers story 17's adaptive-retrieval
gate — skipping the corpus search entirely for questions the model can
answer directly from general knowledge.

`reflect` is the entry point `rag_service.run_rag_with_trace` calls after
each generation, when `enable_self_reflective` is set. A strict-rubric
critic LLM call scores the answer and proposes a sharper question. The
critic always grades against the *original* question, never a previously
refined one — the user's actual ask is what must be satisfied, and a
refined question is only a means to a better retrieval, not a new goal.

`should_regenerate` is the bounded gate `rag_service` calls to decide
whether to loop again. Like CRAG's `relevance_label`, the critic's own
`needs_regeneration` opinion is carried on `ReflectionResult` for logging
only — the actual decision is `reflection_score` against the fixed
`settings.reflection_min_score` threshold, the same threshold-over-label
pattern crag_service already uses, AND'd with a retry ceiling
(`settings.max_reflection_retries`) so a critic that's never satisfied
can't loop forever.

A regenerated answer reruns retrieval on the refined question rather than
reusing the first pass's chunks: the whole point of sharpening the
question is to retrieve better context for it, and a wider question
that failed to retrieve well the first time has no reason to retrieve
better against unchanged chunks.

`needs_retrieval` is the story 17 gate `rag_service` calls before
retrieving at all. A classifier-call outage or malformed response
degrades to "retrieve" — retrieving unnecessarily costs one extra search,
while skipping a search that was actually needed risks an ungrounded
answer, so the safer default under uncertainty is to retrieve. This
mirrors the graceful-degradation contract crag_service, hyde_service, and
reranker_service already follow (story #18).
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import ValidationError

from app.config import settings
from app.models import ReflectionResult, RetrievalDecision, RetrievedChunk
from app.services.chunk_formatting import format_chunks
from app.services.llm_service import generate_json

_NO_CONTEXT_NOTE = "(no context was retrieved — this question was answered from general knowledge)"

_CRITIC_SYSTEM_PROMPT = (
    "You are a strict rubric critic for a Kubernetes operations RAG "
    "assistant. Given a question, the retrieved context, and a generated "
    "answer, judge whether the answer fully and correctly answers the "
    "question and is actually supported by the context — not just "
    "plausible-sounding. If no context was retrieved, the question was "
    "judged general knowledge; judge the answer on whether it's correct "
    "and appropriately confident, not on whether it cites any context.\n\n"
    "If the answer is weak, propose a sharper, more specific rephrasing "
    "of the question that would retrieve better context.\n\n"
    "Respond with a JSON object only, no other text:\n"
    '{"reflection_score": <float 0.0-1.0>, '
    '"needs_regeneration": <bool>, '
    '"refined_question": "<sharper question, or empty if not needed>", '
    '"reasoning": "<one sentence>"}'
)

_RETRIEVAL_GATE_SYSTEM_PROMPT = (
    "You decide whether a question needs a search against a Kubernetes "
    "operations knowledge base, or whether it's general knowledge you can "
    "answer directly, with no lookup, and be confident you're right.\n\n"
    "Respond with a JSON object only, no other text:\n"
    '{"needs_retrieval": <bool>, "reasoning": "<one sentence>"}'
)


def reflect(question: str, answer: str, chunks: list[RetrievedChunk]) -> ReflectionResult:
    """Critique `answer` against the original `question` and the `chunks` it
    was generated from (empty when Self-RAG skipped retrieval — see
    `_NO_CONTEXT_NOTE`). Always returns a `ReflectionResult`; a critic-call
    outage or malformed response degrades to "good enough, don't
    regenerate" — a broken judge isn't evidence the answer is bad."""
    context = format_chunks(chunks) if chunks else _NO_CONTEXT_NOTE
    prompt = f"Question: {question}\n\nRetrieved context:\n{context}\n\nGenerated answer: {answer}"
    try:
        response = generate_json(prompt, system_prompt=_CRITIC_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 — a critic outage degrades to "don't regenerate", never fails the request
        logger.warning("Reflection critic call failed; accepting the answer as-is")
        return ReflectionResult(
            reflection_score=1.0, needs_regeneration=False, reasoning="critic call failed"
        )

    try:
        return ReflectionResult.model_validate(json.loads(response.text))
    except (json.JSONDecodeError, ValidationError):
        logger.warning("Reflection critic returned malformed JSON; accepting the answer as-is")
        return ReflectionResult(
            reflection_score=1.0, needs_regeneration=False, reasoning="critic output was malformed"
        )


def should_regenerate(reflection: ReflectionResult, retries_so_far: int) -> bool:
    """Bounded gate: regenerate only while the score is below
    `settings.reflection_min_score` AND the retry ceiling
    (`settings.max_reflection_retries`) hasn't been reached yet."""
    return (
        reflection.reflection_score < settings.reflection_min_score
        and retries_so_far < settings.max_reflection_retries
    )


def needs_retrieval(question: str) -> bool:
    """Adaptive-retrieval gate (Self-RAG's own "Retrieve" decision, spec.md
    user story 17): general-knowledge questions skip the corpus search
    entirely. Degrades to `True` (retrieve) on any classifier failure."""
    try:
        response = generate_json(question, system_prompt=_RETRIEVAL_GATE_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 — a classifier outage degrades to "retrieve", never fails the request
        logger.warning("Retrieval-gate classifier call failed; defaulting to retrieval")
        return True

    try:
        decision = RetrievalDecision.model_validate(json.loads(response.text))
    except (json.JSONDecodeError, ValidationError):
        logger.warning("Retrieval-gate classifier returned malformed JSON; defaulting to retrieval")
        return True

    return decision.needs_retrieval
