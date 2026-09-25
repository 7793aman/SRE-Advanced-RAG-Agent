"""Cross-encoder reranking: re-score retrieval candidates by actual
query-chunk relevance, not just the embedding/TF-IDF similarity hybrid
search ranked them by.

Pluggable via `settings.reranker_backend`: "local" runs a
sentence-transformers `CrossEncoder` in-process (default; offline after the
model's first download), "voyage" calls Voyage AI's hosted rerank endpoint.
Either backend's failure falls back to the input order, untouched — ticket
#25's own acceptance test, and story #18's graceful-degradation requirement:
a reranker outage degrades to "as if reranking were off", never a failed
request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from app.config import settings
from app.models import RetrievedChunk
from app.services.lazy_singleton import LazySingleton

if TYPE_CHECKING:
    import voyageai
    from sentence_transformers import CrossEncoder


def _build_local_model() -> CrossEncoder:
    # Imported lazily: sentence-transformers pulls in torch, which is slow to
    # import and unneeded in any process that never actually reranks.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


# Two requests racing into a cold cache used to both build this model at
# once — harmless on CPU, but a real deadlock on Apple's MPS backend (same
# failure content_guard.py hit, see issue #34's demo-UI testing). LazySingleton
# serializes the first build instead of racing it.
_local_model: LazySingleton[CrossEncoder] = LazySingleton(_build_local_model)


def _get_local_model() -> CrossEncoder:
    return _local_model.get()


def _build_voyage_client() -> voyageai.Client:
    import voyageai

    return voyageai.Client(api_key=settings.voyage_api_key)


_voyage_client: LazySingleton[voyageai.Client] = LazySingleton(_build_voyage_client)


def _get_voyage_client() -> voyageai.Client:
    return _voyage_client.get()


def _score_local(question: str, texts: list[str]) -> list[float]:
    import torch

    pairs = [(question, text) for text in texts]
    # ms-marco-MiniLM-L-6-v2 (and cross-encoders generally) output raw,
    # unbounded regression logits — a genuinely relevant pair can score -9 —
    # so a sigmoid is required to land in the same [0, 1] relevance range the
    # Voyage backend already returns and `_retrieval_score` assumes.
    #
    # sentence-transformers also types `predict` over a large PairInput union
    # (text, image, audio, video, ...); list's invariance makes our plain
    # list[tuple[str, str]] fail the check even though each pair fits the union.
    scores = _get_local_model().predict(pairs, activation_fn=torch.nn.Sigmoid())  # type: ignore[arg-type]
    return [float(score) for score in scores]


def _score_voyage(question: str, texts: list[str]) -> list[float]:
    result = _get_voyage_client().rerank(question, texts, model=settings.voyage_model)
    scores = [0.0] * len(texts)
    for item in result.results:
        scores[item.index] = item.relevance_score
    return scores


def rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return `chunks` sorted by cross-encoder relevance to `question`,
    most relevant first, with each chunk's `score` updated to that relevance
    score. A single (or empty) input has nothing to reorder, so it's
    returned as-is without invoking the backend."""
    if len(chunks) < 2:
        return chunks

    texts = [chunk.text for chunk in chunks]
    try:
        if settings.reranker_backend == "voyage":
            scores = _score_voyage(question, texts)
        else:
            scores = _score_local(question, texts)
    except Exception:  # noqa: BLE001 — a reranker outage degrades to no-op, never fails the request
        logger.warning("Reranker backend '{}' failed; keeping original order", settings.reranker_backend)
        return chunks

    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [chunk.model_copy(update={"score": score}) for chunk, score in ranked]
