"""Integration seam: vector_store against a live local Qdrant.

Per spec's Testing Decisions, the vector store is tested against a real local
Qdrant, not a mock — same policy as the Postgres tests. Each test runs in its
own throwaway collection (via the qdrant_collection fixture) so tests never
see each other's data and never touch the real `documents` collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.models import RetrievedChunk


def _unit_vector(active_index: int, size: int = 1536) -> list[float]:
    vector = [0.0] * size
    vector[active_index] = 1.0
    return vector


@pytest.fixture(scope="session")
def qdrant_ready() -> None:
    try:
        QdrantClient(url=settings.qdrant_url, timeout=5).get_collections()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip(f"Qdrant not reachable: {exc}")


@pytest.fixture
def qdrant_collection(qdrant_ready: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    name = f"test_vector_store_{uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "qdrant_collection", name)
    yield name
    try:
        QdrantClient(url=settings.qdrant_url).delete_collection(collection_name=name)
    except UnexpectedResponse:
        pass  # already gone, or never created


def test_ensure_collection_creates_it_when_missing(qdrant_collection: str) -> None:
    from app.services.vector_store import ensure_collection, get_client

    ensure_collection()

    existing = {c.name for c in get_client().get_collections().collections}
    assert qdrant_collection in existing


def test_ensure_collection_is_idempotent(qdrant_collection: str) -> None:
    from app.services.vector_store import ensure_collection

    ensure_collection()
    ensure_collection()  # must not raise on the second call


def test_upsert_then_search_returns_the_closest_chunk_first(qdrant_collection: str) -> None:
    from app.services.vector_store import search, upsert_chunks

    chunks = [
        RetrievedChunk(text="pod chunk", source="pods.html"),
        RetrievedChunk(text="deployment chunk", source="deployment.html"),
        RetrievedChunk(text="service chunk", source="service.html"),
    ]
    embeddings = [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
    upsert_chunks(chunks, embeddings)

    results = search(_unit_vector(0), top_k=3)

    assert results[0].text == "pod chunk"
    assert results[0].source == "pods.html"
    assert results[0].score == pytest.approx(1.0)


def test_search_top_k_limits_result_count(qdrant_collection: str) -> None:
    from app.services.vector_store import search, upsert_chunks

    chunks = [RetrievedChunk(text=f"chunk {i}", source="doc.html") for i in range(3)]
    embeddings = [_unit_vector(i) for i in range(3)]
    upsert_chunks(chunks, embeddings)

    results = search(_unit_vector(0), top_k=2)

    assert len(results) == 2


def test_upsert_round_trips_page_number(qdrant_collection: str) -> None:
    from app.services.vector_store import search, upsert_chunks

    chunk_with_page = RetrievedChunk(text="has a page", source="doc.pdf", page_number=7)
    chunk_without_page = RetrievedChunk(text="no page here", source="doc.html")
    upsert_chunks([chunk_with_page, chunk_without_page], [_unit_vector(0), _unit_vector(1)])

    results = search(_unit_vector(0), top_k=1)
    assert results[0].page_number == 7

    results = search(_unit_vector(1), top_k=1)
    assert results[0].page_number is None


def test_fuse_rrf_orders_by_combined_rank() -> None:
    """Hand-calculated with k=60: crash (1/62 + 1/61 = 0.0325) beats rollout
    (1/61 = 0.0164, dense-only) beats service (1/62 = 0.0161, sparse-only) —
    consistently ranking in both lists beats being #1 in only one."""
    from app.services.vector_store import fuse_rrf

    crash = RetrievedChunk(text="crash text", source="crash.html")
    rollout = RetrievedChunk(text="rollout text", source="rollout.html")
    service = RetrievedChunk(text="service text", source="service.html")

    dense_results = [rollout, crash]
    sparse_results = [crash, service]

    fused = fuse_rrf([dense_results, sparse_results], k=60)

    assert [chunk.source for chunk in fused] == ["crash.html", "rollout.html", "service.html"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_sparse_index_on_empty_corpus_does_not_crash(qdrant_collection: str) -> None:
    from app.services.vector_store import build_sparse_index, ensure_collection, sparse_search

    ensure_collection()

    index = build_sparse_index()
    assert index.matrix is None
    assert index.chunks == []

    assert sparse_search("CrashLoopBackOff") == []


def test_sparse_search_finds_exact_token_dense_search_misses(qdrant_collection: str) -> None:
    """Ticket #24's acceptance case: an exact-token query should retrieve the
    right doc via sparse search even when dense search (driven by embedding
    similarity, not word overlap) would surface a different, wrong doc."""
    from app.services.vector_store import search, sparse_search, upsert_chunks

    target = RetrievedChunk(
        text="Pod stuck in CrashLoopBackOff due to imagePullPolicy misconfiguration",
        source="target.html",
    )
    unrelated = RetrievedChunk(
        text="deployment rolling update strategy overview", source="unrelated.html"
    )
    upsert_chunks([target, unrelated], [_unit_vector(0), _unit_vector(1)])

    # Dense search follows the embedding, not the words — a query embedding
    # closest to "unrelated" surfaces the wrong doc regardless of query text.
    dense_results = search(_unit_vector(1), top_k=1)
    assert dense_results[0].source == "unrelated.html"

    # Sparse search follows the exact word "imagePullPolicy" and correctly
    # finds the target doc that dense missed.
    sparse_results = sparse_search("imagePullPolicy", top_k=1)
    assert sparse_results[0].source == "target.html"


def test_source_exists_true_only_for_ingested_file_names(qdrant_collection: str) -> None:
    from app.services.vector_store import source_exists, upsert_chunks

    upsert_chunks(
        [RetrievedChunk(text="pods are the smallest unit", source="pods.html", page_number=None)],
        [_unit_vector(0)],
    )

    assert source_exists("pods.html") is True
    assert source_exists("services.html") is False


def test_source_exists_is_false_before_the_collection_exists(qdrant_collection: str) -> None:
    from app.services.vector_store import source_exists

    assert source_exists("pods.html") is False
