"""`POST /query` — the RAG endpoint: a question in, a grounded chat response out.

Requires a valid JWT (story #28). This endpoint is a thin HTTP wrapper around
`rag_service.run_rag`: unpack the request's feature flags into a dict, run
the pipeline, return its `ChatResponse` as-is. The other security layers
(rate limiting, token budget, injection scanning, PII redaction, ...) are
wired in by ticket #32, not here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.models import ChatResponse, QueryRequest
from app.services.rag_service import run_rag

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
    return run_rag(body.question, _flags(body))
