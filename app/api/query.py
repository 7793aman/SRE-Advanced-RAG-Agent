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

`/query` runs inside the fixed-order security pipeline (ticket #32):

    L1 schema+regex -> L4a JWT -> L4b rate limit -> L6 budget check -> L5 restructure
    -> L2 guard -> L7a redact PII -> graph (L3 prompt + L8 spotlighting inside)
    -> L7b moderate + redact -> L9 validate -> L6 consume

L1 (Pydantic body validation) and L4a (the `get_current_user` dependency) are
resolved by FastAPI before this function body runs, and FastAPI resolves
dependencies before it reports body errors — so a request with a bad token *and*
a malicious body gets the 401, not the 422. That reveals less to an
unauthenticated caller, so we keep it. Everything from L4b down is in `query`
below, in order.

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

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.middleware.rate_limiter import rate_limiter
from app.models import (
    ChatResponse,
    PendingSQLBlock,
    QueryRequest,
    ResponseMetadata,
    SQLExecuteRequest,
)
from app.security.content_guard import moderate_output, scan_input
from app.security.input_restructuring import count_tokens, restructure_input
from app.security.output_validator import validate_output
from app.security.pii_redaction import redact_pii
from app.security.token_budget import token_budget
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


def _clean_response(response: ChatResponse) -> ChatResponse:
    """L7b then L9: moderate and redact the answer, then check its shape."""
    moderate_output(response.answer)
    cleaned = response.model_copy(update={"answer": redact_pii(response.answer)})
    return validate_output(cleaned.model_dump(mode="json"))


@router.post("/query", response_model=ChatResponse)
def query(body: QueryRequest, user: AuthenticatedUser = Depends(get_current_user)) -> ChatResponse:
    if not rate_limiter.is_allowed_user(str(user.id)):  # L4b
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limit_exceeded"
        )
    token_budget.check(user.id)  # L6 (check)
    question = restructure_input(body.question)  # L5
    scan_input(question)  # L2
    question = redact_pii(question)  # L7a

    result = get_graph().invoke(  # L3 + L8 run inside the graph
        {"question": question, "flags": _flags(body), "user_id": user.id},
        {"configurable": {"thread_id": str(uuid4())}},
    )
    raw_response = _to_chat_response(result)

    # L6 (consume) runs in `finally`: the LLM money is already spent once the graph
    # has returned, so an answer L7b or L9 then rejects still counts against the
    # budget. Only a pre-graph rejection (L1/L2/L4b/L6-check) is free. Charged on
    # the raw answer, not the redacted one — redaction doesn't change token cost.
    # Estimate: the question plus the answer. The graph makes several LLM calls
    # (routing, grading, ...) whose usage it doesn't report back, so this
    # undercounts real spend; it's a per-user bound, not an invoice.
    try:
        response = _clean_response(raw_response)  # L7b + L9
    finally:
        token_budget.consume(user.id, count_tokens(question) + count_tokens(raw_response.answer))
    return response


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
