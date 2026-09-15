"""Unit seam: web_search(), with the Tavily client faked. Nothing here calls
the real Tavily API.

Unlike reranker_service / hyde_service, a missing API key is not a silent
no-op here: `WebSearchUnconfiguredError` is raised explicitly (spec.md's web
search service: "explicit error when unconfigured"). It's `crag_service` that
catches this and degrades — see test_crag_service.py.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.models import RetrievedChunk


class _FakeTavilyClient:
    """Records every call it's asked to make; returns a fixed result set."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.calls: list[dict] = []

    def search(self, query: str, **kwargs: object) -> dict:
        self.calls.append({"query": query, **kwargs})
        return {"results": self._results}


@pytest.fixture(autouse=True)
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", "test-tavily-key")


def test_raises_explicitly_when_api_key_is_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import web_search_service

    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(
        web_search_service,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("client should never be built")),
    )

    with pytest.raises(web_search_service.WebSearchUnconfiguredError):
        web_search_service.web_search("latest stable Kubernetes release")


def test_returns_results_as_retrieved_chunks_with_the_url_as_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import web_search_service

    fake_client = _FakeTavilyClient(
        [
            {
                "url": "https://kubernetes.io/releases/",
                "content": "Kubernetes v1.32 is the latest stable release.",
                "title": "Kubernetes Releases",
                "score": 0.87,
            }
        ]
    )
    monkeypatch.setattr(web_search_service, "_get_client", lambda: fake_client)

    result = web_search_service.web_search("latest stable Kubernetes release")

    assert result == [
        RetrievedChunk(
            text="Kubernetes v1.32 is the latest stable release.",
            source="https://kubernetes.io/releases/",
            score=0.87,
        )
    ]


def test_passes_the_query_and_max_results_through_to_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import web_search_service

    fake_client = _FakeTavilyClient([])
    monkeypatch.setattr(web_search_service, "_get_client", lambda: fake_client)

    web_search_service.web_search("latest stable Kubernetes release", max_results=3)

    assert fake_client.calls == [{"query": "latest stable Kubernetes release", "max_results": 3}]


def test_no_results_returns_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import web_search_service

    monkeypatch.setattr(web_search_service, "_get_client", lambda: _FakeTavilyClient([]))

    assert web_search_service.web_search("a question with no web hits") == []


def test_missing_score_field_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import web_search_service

    fake_client = _FakeTavilyClient(
        [{"url": "https://example.com/a", "content": "some content", "title": "A"}]
    )
    monkeypatch.setattr(web_search_service, "_get_client", lambda: fake_client)

    result = web_search_service.web_search("a question")

    assert result[0].score == 0.0
