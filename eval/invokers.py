"""Calls the real app so the eval harness asks it questions the same way a real
user would — `ServiceInvoker` goes straight to the uncached RAG service function,
bypassing HTTP/auth/the graph's SQL-approval interrupt entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import settings
from app.models import ChatResponse, RetrievedChunk
from app.services.rag_service import run_rag_with_trace


class SkippedIntent(Exception):
    pass


class Invoker(ABC):
    @abstractmethod
    def invoke(
        self, question: str, flags: dict, intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]: ...


class ServiceInvoker(Invoker):
    """Calls `run_rag_with_trace` directly. Only `rag` and `web_fallback` goldens
    can run this way — `sql` and `hybrid` need the compiled graph's SQL-approval
    interrupt and a real Postgres ops DB, which means going through the actual
    `/query` + `/query/sql/execute` HTTP endpoints with a JWT. That's real work
    (an "api" invocation mode), not yet built here — sql/hybrid_rag_sql goldens
    are skipped, not silently scored as failures, until it is.
    """

    SUPPORTED_INTENTS = {"rag", "web_fallback"}

    def invoke(
        self, question: str, flags: dict, intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        if intent not in self.SUPPORTED_INTENTS:
            raise SkippedIntent(f"intent={intent} not supported in service mode")

        if intent == "web_fallback" and not settings.tavily_api_key:
            raise SkippedIntent("tavily_unset: TAVILY_API_KEY not configured")

        return run_rag_with_trace(question, flags)
