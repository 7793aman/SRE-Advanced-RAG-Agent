"""`POST /query` — the RAG endpoint: a question in, a grounded chat response out.

Requires a valid JWT (story #28). This endpoint is a thin HTTP wrapper around
the compiled LangGraph state machine (issue #29; `app.services.graph`):
unpack the request's feature flags into a dict, invoke the graph with a
fresh thread id, return the `ChatResponse` it produced. The other security
layers (rate limiting, token budget, injection scanning, PII redaction, ...)
are wired in by ticket #32, not here.

A new `thread_id` per request is the right default today: nothing here
resumes a previous run. The SQL-approval ticket that follows #29 will need
a *stable* thread id (to resume a paused graph across a second HTTP
request) — swap it out then, not before.

Known follow-up, not addressed here: every request writes a permanent
checkpoint row under its own thread id, with nothing in this codebase to
ever delete old ones — the checkpoint tables grow without bound under
sustained traffic. Retention (a TTL, a cleanup job, or reusing threads)
is an operational concern with no acceptance criterion in issue #29;
tracking it here rather than adding untested cleanup logic no ticket asked
for.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.models import ChatResponse, QueryRequest
from app.services.graph import get_graph

router = APIRouter(tags=["query"])


def _flags(body: QueryRequest) -> dict[str, object]:
    return {
        "search_mode": body.search_mode,
        "enable_rerank": body.enable_rerank,
        "enable_hyde": body.enable_hyde,
        "enable_crag": body.enable_crag,
        "enable_self_reflective": body.enable_self_reflective,
        "enable_adaptive_retrieval": body.enable_adaptive_retrieval,
        "top_k": body.top_k,
    }


@router.post("/query", response_model=ChatResponse)
def query(body: QueryRequest, user: AuthenticatedUser = Depends(get_current_user)) -> ChatResponse:
    result = get_graph().invoke(
        {"question": body.question, "flags": _flags(body)},
        {"configurable": {"thread_id": str(uuid4())}},
    )
    return result["response"]
