"""`POST /query` and `POST /query/sql/execute` — the RAG / Text2SQL endpoints.

Requires a valid JWT (story #28). These endpoints are thin HTTP wrappers
around the compiled LangGraph state machine (issues #29, #30;
`app.services.graph`): unpack the request's feature flags into a dict,
invoke the graph with a fresh thread id, return the `ChatResponse` it
produced.

A SQL-intent question pauses the graph for human approval. `/query` then
returns `pending_sql` (with no answer) and the thread id as its `query_id`;
`/query/sql/execute` resumes that same thread with the human's decision.
A resume is only accepted for a run that is actually paused at the approval
step and was started by the same user — anything else is a 404, so a query
can be approved once, by its owner, and its id reveals nothing to others.

The other security layers (rate limiting, token budget, injection scanning,
PII redaction, ...) are wired in by ticket #32, not here.

Known follow-up, not addressed here: every request writes a permanent
checkpoint row under its own thread id, with nothing in this codebase to
ever delete old ones — the checkpoint tables grow without bound under
sustained traffic (and a never-answered approval stays paused forever).
Retention (a TTL, a cleanup job, or reusing threads) is an operational
concern with no acceptance criterion in issue #29 or #30; tracking it here
rather than adding untested cleanup logic no ticket asked for.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.models import (
    ChatResponse,
    PendingSQLBlock,
    QueryRequest,
    ResponseMetadata,
    SQLExecuteRequest,
)
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


def _to_chat_response(result: dict[str, Any]) -> ChatResponse:
    """The graph either finished (`response`) or paused for SQL approval."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        return ChatResponse(
            answer="",
            retrieval_score=0.0,
            pending_sql=PendingSQLBlock(**interrupts[0].value),
            metadata=ResponseMetadata(route="sql_pending"),
        )
    return result["response"]


@router.post("/query", response_model=ChatResponse)
def query(body: QueryRequest, user: AuthenticatedUser = Depends(get_current_user)) -> ChatResponse:
    result = get_graph().invoke(
        {"question": body.question, "flags": _flags(body), "user_id": user.id},
        {"configurable": {"thread_id": str(uuid4())}},
    )
    return _to_chat_response(result)


@router.post("/query/sql/execute", response_model=ChatResponse)
def execute_sql_query(
    body: SQLExecuteRequest, user: AuthenticatedUser = Depends(get_current_user)
) -> ChatResponse:
    graph = get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": body.query_id}}

    snapshot = graph.get_state(config)
    is_paused = snapshot.next == ("request_sql_approval",)
    if not is_paused or snapshot.values.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="No pending SQL query with that id")

    return _to_chat_response(graph.invoke(Command(resume=body.approved), config))
