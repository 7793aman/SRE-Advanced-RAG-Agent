"""Unit seam: rerank(), with the local and hosted scoring backends faked.

Nothing here loads the real sentence-transformers model or calls Voyage —
each backend's scoring function is faked so we can check ordering and
failure-fallback behaviour without a model download or an API call.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.models import RetrievedChunk

_CHUNKS = [
    RetrievedChunk(text="A Deployment manages replica Pods.", source="deployments.html", score=0.6),
    RetrievedChunk(text="OOMKilled means the kernel killed the container for exceeding its memory limit.", source="oom.html", score=0.4),
    RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=0.5),
]


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "reranker_backend", "local")


def test_reorders_chunks_by_relevance_score_most_relevant_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reranker_service

    # The gold chunk (oom.html) is last going in but scores highest.
    monkeypatch.setattr(reranker_service, "_score_local", lambda q, texts: [0.2, 0.9, 0.1])

    result = reranker_service.rerank("Why is my pod OOMKilled?", _CHUNKS)

    assert [chunk.source for chunk in result] == ["oom.html", "deployments.html", "pods.html"]


def test_reranked_chunks_carry_the_new_relevance_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import reranker_service

    monkeypatch.setattr(reranker_service, "_score_local", lambda q, texts: [0.2, 0.9, 0.1])

    result = reranker_service.rerank("Why is my pod OOMKilled?", _CHUNKS)

    assert result[0].score == pytest.approx(0.9)


def test_returns_input_order_unchanged_on_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reranker_service

    def _boom(question: str, texts: list[str]) -> list[float]:
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(reranker_service, "_score_local", _boom)

    result = reranker_service.rerank("Why is my pod OOMKilled?", _CHUNKS)

    assert result == _CHUNKS


def test_uses_the_voyage_backend_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import reranker_service

    monkeypatch.setattr(settings, "reranker_backend", "voyage")
    monkeypatch.setattr(
        reranker_service, "_score_local", lambda q, texts: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(reranker_service, "_score_voyage", lambda q, texts: [0.9, 0.1, 0.5])

    result = reranker_service.rerank("Why is my pod OOMKilled?", _CHUNKS)

    assert result[0].source == "deployments.html"


def test_fewer_than_two_chunks_is_returned_as_is_without_calling_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reranker_service

    monkeypatch.setattr(
        reranker_service, "_score_local", lambda q, texts: (_ for _ in ()).throw(AssertionError())
    )

    assert reranker_service.rerank("question", []) == []
    assert reranker_service.rerank("question", [_CHUNKS[0]]) == [_CHUNKS[0]]
