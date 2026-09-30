"""Langfuse tracing, behind a wrapper that does nothing until it is configured.

Tracing follows the same degrade-don't-die rule as the reranker, web search and
cache: with `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` unset, every helper
here is a pass-through no-op — no client is built, no network call is made, and
nothing new is required in `.env`. A Langfuse failure while enabled is logged
and swallowed too; a tracing outage must never fail a `/query`.

The rest of the app imports only this module, never `langfuse` directly:

  - `observe`            decorator; one span (or `as_type="generation"`) per call
  - `update_generation`  attach model + token usage to the current LLM span
  - `update_trace`       set trace input/output; metadata (flags, chunks) lands on the current span
  - `callbacks`          LangChain/LangGraph callback handlers for `graph.invoke`
  - `current_trace_id`   so the eval harness can link a score to its trace
  - `get_client`         the raw client (None when disabled), for datasets/scores
  - `flush`              send buffered events (short-lived processes, shutdown)

Redaction: every payload goes through `mask` — the same L7 regex redaction the
API applies to questions and answers — inside the SDK, before it exports
anything. Retrieved chunk text is the payload most likely to hold private data
the user never typed, so it is masked too.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel

from app.config import settings
from app.security.pii_redaction import redact_pii
from app.services.lazy_singleton import LazySingleton


def is_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def mask(data: Any, **_: Any) -> Any:
    """Redact PII from every string inside a trace payload. Never raises: the
    SDK would drop the whole payload on an error, and an unmasked one must not
    slip through either, so an unmaskable value is replaced, not passed on."""
    try:
        if isinstance(data, str):
            return redact_pii(data)
        if isinstance(data, dict):
            return {key: mask(value) for key, value in data.items()}
        if isinstance(data, list | tuple | set | frozenset):
            return [mask(item) for item in data]
        if isinstance(data, BaseModel):
            return mask(data.model_dump(mode="json"))
    except Exception:  # noqa: BLE001
        return "[UNMASKABLE]"
    return data


def _build_client() -> Any:
    from langfuse import Langfuse

    # Pass the keys explicitly: pydantic-settings reads `.env` without exporting
    # it to `os.environ`, which is where the SDK would otherwise look.
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url or settings.langfuse_host or None,
        mask=mask,
    )


_client: LazySingleton[Any] = LazySingleton(_build_client)


def _get_client() -> Any:
    return _client.get()


_RETRY_BUILD_AFTER_SECONDS = 60.0
_build_failed_at: float | None = None


def get_client() -> Any | None:
    """The Langfuse client, or None when tracing is off or the client won't build.

    After a failed build, don't retry (or log) on every call — wait a minute."""
    global _build_failed_at
    if not is_enabled():
        return None
    if _build_failed_at is not None:
        if time.monotonic() - _build_failed_at < _RETRY_BUILD_AFTER_SECONDS:
            return None
        _build_failed_at = None
    try:
        return _get_client()
    except Exception:  # noqa: BLE001
        _build_failed_at = time.monotonic()
        logger.warning("Langfuse client could not be built; tracing skipped")
        return None


def observe(
    name: str | None = None, as_type: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Trace a function as a Langfuse span. A plain call while tracing is off."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        traced: Callable[..., Any] | None = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced
            if get_client() is None:
                return fn(*args, **kwargs)
            if traced is None:
                try:
                    from langfuse import observe as lf_observe

                    traced = lf_observe(name=name or fn.__name__, as_type=as_type)(fn)  # type: ignore[arg-type,call-overload]
                except Exception:  # noqa: BLE001
                    logger.warning("Langfuse observe unavailable; running untraced")
                    return fn(*args, **kwargs)
            return traced(*args, **kwargs)

        return wrapper

    return decorator


def update_generation(
    *,
    model: str | None = None,
    usage: dict[str, int] | None = None,
    input: Any = None,  # noqa: A002
    output: Any = None,
) -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.update_current_generation(
            model=model,
            usage_details=usage,
            input=input,
            output=output,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse generation update failed")


def update_trace(
    *,
    input: Any = None,  # noqa: A002
    output: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    client = get_client()
    if client is None:
        return
    try:
        if input is not None or output is not None:
            client.set_current_trace_io(input=input, output=output)
        if metadata is not None:
            client.update_current_span(metadata=metadata)
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse trace update failed")


def callbacks() -> list[Any]:
    """Callback handlers to put in a LangGraph `config["callbacks"]`."""
    if get_client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse callback handler unavailable")
        return []


def current_trace_id() -> str | None:
    client = get_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:  # noqa: BLE001
        return None


def flush() -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse flush failed")
