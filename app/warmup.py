"""Startup warm-up: load the slow-to-build pieces before the first question.

Each piece below is built lazily on first use — the llm-guard scanner models,
the TF-IDF keyword index, the local reranker, the LangGraph graph — so without
this the first question after a restart pays for all of them at once.

Runs in a background thread so the server accepts requests immediately. Every
step is independent and best-effort: a failure is logged and skipped, and the
piece just loads lazily on first use like before.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from app.config import settings

logger = logging.getLogger(__name__)


def _warm_input_scanners() -> None:
    from app.security.content_guard import _load_input_scanners

    _load_input_scanners()


def _warm_output_scanners() -> None:
    from app.security.content_guard import _load_output_scanners

    _load_output_scanners()


def _warm_sparse_index() -> None:
    from app.services.vector_store import _fit_sparse_index

    _fit_sparse_index(settings.qdrant_collection)


def _warm_reranker() -> None:
    if settings.reranker_backend != "local":
        return
    from app.services.reranker_service import _get_local_model

    _get_local_model()


def _warm_graph() -> None:
    from app.services.graph import get_graph

    get_graph()


_STEPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("input scanners", _warm_input_scanners),
    ("output scanners", _warm_output_scanners),
    ("keyword index", _warm_sparse_index),
    ("reranker", _warm_reranker),
    ("graph", _warm_graph),
)


def run_warmup() -> None:
    for name, step in _STEPS:
        started = time.monotonic()
        try:
            step()
        except Exception:
            logger.warning("warm-up: %s failed; will load on first use", name, exc_info=True)
        else:
            logger.info("warm-up: %s ready in %.1fs", name, time.monotonic() - started)


def start_warmup() -> None:
    if not settings.warmup_on_startup:
        return
    threading.Thread(target=run_warmup, name="warmup", daemon=True).start()
