"""The intent router: classifies each question as `rag` / `sql` / `hybrid`
(issue #29; spec.md user story 2, "Router service") so the graph
(`app/services/graph.py`) knows which evidence family a question needs —
the document corpus, the operational Postgres database, or both. Cached by
the existing `intent` tier (`query_cache_service.py`) so a repeated
question skips the classifier call entirely.

Degrades to `rag` on any classifier failure or unexpected output, the same
graceful-degradation contract `reflection_service.needs_retrieval` follows
— a classifier outage shouldn't fail the request, and `rag` is the only
intent this repo can actually execute today (the `sql` and `hybrid` graph
paths are a later ticket). Unlike a degraded `needs_retrieval` call,
though, a degraded classification is deliberately *not* cached: caching it
would stick a wrong classification for a full day (`cache_ttl_intent`)
even after the classifier itself recovers.
"""

from __future__ import annotations

import json

from loguru import logger
from pydantic import ValidationError

from app.models import IntentClassification
from app.services.llm_service import generate_json
from app.services.query_cache_service import query_cache

_ROUTER_SYSTEM_PROMPT = (
    "You route Kubernetes operations questions for a system with two evidence "
    "sources: a documentation corpus (Kubernetes concepts and operations "
    "guides) and an operational Postgres database (clusters, nodes, "
    "deployments, pods, incidents, alerts, oncall_logs).\n\n"
    "Classify the question as exactly one of:\n"
    '- "rag" - answerable from the documentation alone, e.g. "how does a '
    'Deployment roll out?"\n'
    '- "sql" - answerable from the operational database alone, e.g. "how many '
    'P1 incidents happened last month?"\n'
    '- "hybrid" - needs documentation and database evidence together, e.g. '
    '"list P1 incidents and their recommended remediation steps"\n\n'
    "Respond with a JSON object only, no other text:\n"
    '{"intent": "rag"|"sql"|"hybrid", "reasoning": "<one sentence>"}'
)


def classify_intent(question: str) -> str:
    """Return the cached intent for `question`, classifying and caching it
    on a miss. Always returns one of `rag` / `sql` / `hybrid`."""
    cached = query_cache.get_intent(question)
    if cached is not None:
        return cached

    classification = _classify(question)
    if classification is None:
        return "rag"

    query_cache.set_intent(question, classification)
    return classification


def _classify(question: str) -> str | None:
    """Ask the LLM to classify `question`. Returns `None` (do not cache) on
    any classifier failure or malformed/unexpected output."""
    try:
        response = generate_json(question, system_prompt=_ROUTER_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 — a router outage degrades to rag, never fails the request
        logger.warning("Intent router call failed; defaulting to rag")
        return None

    try:
        return IntentClassification.model_validate(json.loads(response.text)).intent
    except (json.JSONDecodeError, ValidationError):
        logger.warning("Intent router returned malformed output; defaulting to rag")
        return None
