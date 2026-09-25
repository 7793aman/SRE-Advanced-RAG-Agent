"""Tavily web search: CRAG's fallback source when corpus retrieval grades too
low to trust (spec.md user story 15; "Web search service").

Every other optional backend in this codebase (reranker_service, hyde_service)
degrades a missing dependency into a silent no-op. This one doesn't: a missing
`TAVILY_API_KEY` raises `WebSearchUnconfiguredError` explicitly, because a
misconfigured web-fallback should be loud at the layer closest to its cause,
not invisible. `crag_service` is the layer one level up that catches this
error and degrades the request without crashing it — see that module.

Results come back as `RetrievedChunk`s with the source URL in the `source`
field, the same field a corpus chunk's filename lives in — so spotlighting,
prompt-building, and source citation downstream all work unchanged.
"""

from __future__ import annotations

from tavily import TavilyClient

from app.config import settings
from app.models import RetrievedChunk
from app.services.lazy_singleton import LazySingleton


class WebSearchUnconfiguredError(RuntimeError):
    """Raised when `web_search` is called without `TAVILY_API_KEY` set."""


_client: LazySingleton[TavilyClient] = LazySingleton(
    lambda: TavilyClient(api_key=settings.tavily_api_key)
)


def _get_client() -> TavilyClient:
    return _client.get()


def web_search(query: str, max_results: int = 5) -> list[RetrievedChunk]:
    """Search the live web via Tavily and return results as `RetrievedChunk`s,
    most relevant first, each `source` set to its URL.

    Raises `WebSearchUnconfiguredError` when `TAVILY_API_KEY` isn't set —
    see the module docstring for why this is explicit rather than a silent
    empty result.
    """
    if not settings.tavily_api_key:
        raise WebSearchUnconfiguredError("TAVILY_API_KEY is not configured")

    response = _get_client().search(query, max_results=max_results)
    return [
        RetrievedChunk(
            text=result["content"],
            source=result["url"],
            score=float(result.get("score", 0.0)),
        )
        for result in response["results"]
    ]
