"""Batched OpenAI embeddings with per-text cache read-through.

Each text is looked up in the `embedding` cache tier before any OpenAI call.
Only the misses are sent to OpenAI, batched into one request, and the results
are written back to the cache — so a repeated `embed_texts` call for text
already seen is a pure cache hit with no OpenAI call at all.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.services.query_cache_service import query_cache


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    resolved_model = model or settings.embedding_model

    vectors: dict[int, list[float]] = {}
    miss_indices: list[int] = []
    for i, text in enumerate(texts):
        cached = query_cache.get_embedding(text, resolved_model)
        if cached is not None:
            vectors[i] = cached
        else:
            miss_indices.append(i)

    if miss_indices:
        # dict.fromkeys dedupes while keeping first-occurrence order, so a
        # chunk repeated within the same call (e.g. a boilerplate header) is
        # only sent to OpenAI once.
        unique_miss_texts = list(dict.fromkeys(texts[i] for i in miss_indices))
        response = _get_client().embeddings.create(input=unique_miss_texts, model=resolved_model)
        vector_by_text = {
            text: item.embedding
            for text, item in zip(unique_miss_texts, response.data, strict=True)
        }
        for i in miss_indices:
            vector = vector_by_text[texts[i]]
            vectors[i] = vector
            query_cache.set_embedding(texts[i], vector, resolved_model)

    return [vectors[i] for i in range(len(texts))]
