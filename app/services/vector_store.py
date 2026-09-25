"""Qdrant-backed vector store: collection lifecycle, upsert, dense search;
plus an in-process TF-IDF sparse index built by scrolling the same collection.

One collection (`settings.qdrant_collection`), cosine distance, sized for
`text-embedding-3-small` (1536 dims). `ensure_collection()` is idempotent and
called before every write, so a caller never has to think about whether the
collection already exists before storing chunks in it. `search()` doesn't
call it — by the time anything searches, ingestion (`upsert_chunks`, via
`make seed`) is expected to have already created the collection.

The sparse index is built once per collection and cached in memory for the
life of the process (`build_sparse_index()` -> `_fit_sparse_index()`, cached
by collection name). It goes stale if you seed new documents into a
collection a running process already has cached — a server restart (or
reseeding into a fresh collection, as tests do) picks up the change. That's
the same restart-to-refresh trade you'd accept for any in-process cache; it
just doesn't apply to dense search because Qdrant's own index lives outside
this process and updates itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.models import RetrievedChunk

VECTOR_SIZE = 1536
SCROLL_PAGE_SIZE = 256


@dataclass
class SparseIndex:
    """An in-process TF-IDF fit over the current contents of the collection.

    `matrix` is `None` when the collection is empty — there's nothing to fit
    a vocabulary on, so callers must check for that before calling
    `vectorizer.transform(...)`.
    """

    vectorizer: TfidfVectorizer
    matrix: spmatrix | None
    chunks: list[RetrievedChunk]


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
    ensure_collection()
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
            },
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    get_client().upsert(collection_name=settings.qdrant_collection, points=points)
    _fit_sparse_index.cache_clear()


def source_exists(source: str) -> bool:
    """True if the collection already holds a chunk from the file named `source`.
    Used by ingestion to skip files it has already embedded."""
    client = get_client()
    if settings.qdrant_collection not in {c.name for c in client.get_collections().collections}:
        return False
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        limit=1,
    )
    return bool(points)


def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    results = (
        get_client()
        .query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        .points
    )

    return [
        RetrievedChunk(
            text=point.payload.get("text", "") if point.payload else "",
            source=point.payload.get("source", "") if point.payload else "",
            page_number=point.payload.get("page_number") if point.payload else None,
            score=float(point.score),
        )
        for point in results
    ]


@cache
def _fit_sparse_index(collection: str) -> SparseIndex:
    """Scroll every point in `collection` and fit a TF-IDF vectorizer over its
    text. Cached per collection name, so this only runs once per collection
    per process — not once per search."""
    chunks: list[RetrievedChunk] = []
    offset = None
    while True:
        points, offset = get_client().scroll(
            collection_name=collection,
            with_payload=True,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
        )
        chunks.extend(
            RetrievedChunk(
                text=point.payload.get("text", "") if point.payload else "",
                source=point.payload.get("source", "") if point.payload else "",
                page_number=point.payload.get("page_number") if point.payload else None,
            )
            for point in points
        )
        if offset is None:
            break

    vectorizer = TfidfVectorizer()
    if not chunks:
        return SparseIndex(vectorizer=vectorizer, matrix=None, chunks=chunks)

    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return SparseIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def build_sparse_index() -> SparseIndex:
    """The in-process TF-IDF index for the current collection. Built once per
    collection name and reused after that — see `_fit_sparse_index`."""
    return _fit_sparse_index(settings.qdrant_collection)


def sparse_search(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    index = build_sparse_index()
    if index.matrix is None:
        return []

    query_vector = index.vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, index.matrix)[0]
    # A zero score means the chunk shares no words with the query at all —
    # padding the result with these would inject unrelated noise into
    # hybrid_search's RRF input rather than just returning fewer results.
    ranked_positions = [
        position
        for position in similarities.argsort()[::-1][:top_k]
        if similarities[position] > 0
    ]

    return [
        RetrievedChunk(
            text=index.chunks[position].text,
            source=index.chunks[position].source,
            page_number=index.chunks[position].page_number,
            score=float(similarities[position]),
        )
        for position in ranked_positions
    ]


def fuse_rrf(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Merge several already-ranked chunk lists (e.g. dense + sparse) into one,
    using Reciprocal Rank Fusion: sum 1/(k + rank) per chunk across every list
    it appears in, then sort by that total. A chunk is matched across lists by
    (source, page_number, text) — RetrievedChunk has no separate id."""
    scores: dict[tuple[str, int | None, str], float] = {}
    first_seen: dict[tuple[str, int | None, str], RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            key = (chunk.source, chunk.page_number, chunk.text)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            first_seen.setdefault(key, chunk)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        RetrievedChunk(
            text=first_seen[key].text,
            source=first_seen[key].source,
            page_number=first_seen[key].page_number,
            score=scores[key],
        )
        for key in ranked_keys
    ]


_CANDIDATE_MULTIPLIER = 2


def hybrid_search(query: str, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    """Run dense and sparse search independently, then fuse them with RRF.

    Each side is asked for more than `top_k` candidates before fusing —
    otherwise a chunk that dense ranks just outside `top_k` (but sparse ranks
    #1) never reaches `fuse_rrf` at all, undercutting hybrid's "at least as
    good as the better of the two" guarantee."""
    candidate_k = top_k * _CANDIDATE_MULTIPLIER
    dense_results = search(query_embedding, top_k=candidate_k)
    sparse_results = sparse_search(query, top_k=candidate_k)
    fused = fuse_rrf([dense_results, sparse_results])
    return fused[:top_k]
