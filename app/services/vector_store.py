"""Qdrant-backed dense vector store: collection lifecycle, upsert, dense search.

One collection (`settings.qdrant_collection`), cosine distance, sized for
`text-embedding-3-small` (1536 dims). `ensure_collection()` is idempotent and
called before every write, so a caller never has to think about whether the
collection already exists before storing chunks in it. `search()` doesn't
call it — by the time anything searches, ingestion (`upsert_chunks`, via
`make seed`) is expected to have already created the collection.
"""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.models import RetrievedChunk

VECTOR_SIZE = 1536


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
