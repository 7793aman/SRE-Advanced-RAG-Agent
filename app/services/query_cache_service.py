"""Multi-tier query cache: embedding, intent, rag_answer, sql_gen, sql_result.

One key-value backend, chosen once at construction — real Upstash Redis when
configured, an in-process TTL dict otherwise. Mirrors the Redis-or-memory
pattern in `app/middleware/rate_limiter.py`: a small `Protocol` the two
backends satisfy, picked by `_default_backend()`, swappable in tests without
touching real Redis.

Every key is `{tier}:{sha256(normalised input)}`. Each tier tracks its own
hits/misses/sets so `stats()` can report a per-tier hit rate.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Protocol

from app.config import settings

_TIERS = ("embedding", "intent", "rag_answer", "sql_gen", "sql_result")


class _KVBackend(Protocol):
    """The slice of key-value ops the cache needs."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    def delete(self, keys: list[str]) -> None: ...


class MemoryBackend:
    """In-process TTL dict. Per-process only — fine for local dev and tests."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.time():
            self._store.pop(key, None)  # pop, not del: two threads may both expire this key
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)

    def delete(self, keys: list[str]) -> None:
        for key in keys:
            self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


class _UpstashBackend:
    """Adapter over `upstash_redis.Redis` — plain string get/set/delete."""

    def __init__(self, url: str, token: str) -> None:
        from upstash_redis import Redis

        self._redis = Redis(url=url, token=token)

    def get(self, key: str) -> str | None:
        value = self._redis.get(key)
        return None if value is None else str(value)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._redis.set(key, value, ex=ttl_seconds)

    def delete(self, keys: list[str]) -> None:
        if keys:
            self._redis.delete(*keys)


def _default_backend() -> _KVBackend:
    if settings.upstash_redis_url and settings.upstash_redis_token:
        return _UpstashBackend(settings.upstash_redis_url, settings.upstash_redis_token)
    return MemoryBackend()


_TTL_BY_TIER: dict[str, int] = {
    "embedding": settings.cache_ttl_embeddings,
    "intent": settings.cache_ttl_intent,
    "rag_answer": settings.cache_ttl_rag,
    "sql_gen": settings.cache_ttl_sql_gen,
    "sql_result": settings.cache_ttl_sql_result,
}


class QueryCacheService:
    def __init__(self, backend: _KVBackend | None = None) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._stats: dict[str, dict[str, int]] = {
            tier: {"hits": 0, "misses": 0, "sets": 0} for tier in _TIERS
        }
        # Keys ever written per tier, so `clear()` can delete them by name even
        # against a Redis backend that doesn't support SCAN-by-pattern well.
        self._keys_by_tier: dict[str, set[str]] = {tier: set() for tier in _TIERS}
        self._lock = threading.Lock()

    def _key(self, tier: str, raw: str) -> str:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{tier}:{digest}"

    def _get(self, tier: str, key: str) -> str | None:
        value = self._backend.get(key)
        with self._lock:
            self._stats[tier]["hits" if value is not None else "misses"] += 1
        return value

    def _set(self, tier: str, key: str, value: str) -> None:
        self._backend.set(key, value, _TTL_BY_TIER[tier])
        with self._lock:
            self._stats[tier]["sets"] += 1
            self._keys_by_tier[tier].add(key)

    # --- embedding tier -----------------------------------------------------

    def embedding_key(self, text: str, model: str) -> str:
        # The model is part of the key: two models produce different vectors
        # (and often different dimensions) for the same text.
        return self._key("embedding", f"{model}:{text}")

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        value = self._get("embedding", self.embedding_key(text, model))
        return None if value is None else json.loads(value)

    def set_embedding(self, text: str, vector: list[float], model: str) -> None:
        self._set("embedding", self.embedding_key(text, model), json.dumps(vector))

    # --- intent tier ---------------------------------------------------------

    def intent_key(self, question: str) -> str:
        return self._key("intent", question.strip().lower())

    def get_intent(self, question: str) -> str | None:
        return self._get("intent", self.intent_key(question))

    def set_intent(self, question: str, intent: str) -> None:
        self._set("intent", self.intent_key(question), intent)

    # --- rag_answer tier -------------------------------------------------------
    # Story #42: the key must include the feature flags, so toggling a
    # technique (rerank, HyDE, ...) can't return another profile's stale answer.

    def rag_answer_key(self, question: str, flags: dict[str, Any] | None = None) -> str:
        payload = {"question": question.strip(), "flags": flags or {}}
        return self._key("rag_answer", json.dumps(payload, sort_keys=True))

    def get_rag_answer(
        self, question: str, flags: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        value = self._get("rag_answer", self.rag_answer_key(question, flags))
        return None if value is None else json.loads(value)

    def set_rag_answer(
        self,
        question: str,
        answer_payload: dict[str, Any],
        flags: dict[str, Any] | None = None,
    ) -> None:
        self._set("rag_answer", self.rag_answer_key(question, flags), json.dumps(answer_payload))

    # --- sql_gen tier ----------------------------------------------------------

    def sql_gen_key(self, question: str) -> str:
        return self._key("sql_gen", question.strip().lower())

    def get_sql_generation(self, question: str) -> str | None:
        return self._get("sql_gen", self.sql_gen_key(question))

    def set_sql_generation(self, question: str, sql: str) -> None:
        self._set("sql_gen", self.sql_gen_key(question), sql)

    # --- sql_result tier -------------------------------------------------------

    def sql_result_key(self, sql: str) -> str:
        return self._key("sql_result", " ".join(sql.split()).strip().lower())

    def get_sql_result(self, sql: str) -> list[dict[str, Any]] | None:
        value = self._get("sql_result", self.sql_result_key(sql))
        return None if value is None else json.loads(value)

    def set_sql_result(self, sql: str, rows: list[dict[str, Any]]) -> None:
        self._set("sql_result", self.sql_result_key(sql), json.dumps(rows))

    # --- admin ---------------------------------------------------------------

    def ping(self) -> bool:
        """A real round-trip against the active backend, for `/admin/health`.

        Bypasses `_get`/`_set` (and their tier stats) — this is a liveness
        probe, not a cached lookup, so it shouldn't show up as a hit or a
        miss in `stats()`. Always `True` for the in-process fallback (story
        #45: it's never actually down); a real `False` only when Redis is
        configured and unreachable.
        """
        try:
            self._backend.set("__healthcheck__", "1", ttl_seconds=5)
            return self._backend.get("__healthcheck__") == "1"
        except Exception:  # noqa: BLE001 — a health probe reports down, never raises
            return False

    def stats(self) -> dict[str, dict[str, float | int]]:
        """Per-tier hits/misses/sets and hit rate, for `/admin/cache/stats`."""
        with self._lock:
            snapshot: dict[str, dict[str, float | int]] = {}
            for tier in _TIERS:
                counts = self._stats[tier]
                total = counts["hits"] + counts["misses"]
                snapshot[tier] = {
                    **counts,
                    "hit_rate": (counts["hits"] / total) if total else 0.0,
                }
            return snapshot

    def clear(self) -> None:
        """Drop every cached entry and reset stats, for `/admin/cache/clear`.

        The (potentially slow, network) backend delete happens outside the
        lock — only the in-process bookkeeping reset needs it — so a clear
        against real Redis doesn't stall every other request's stats update
        for the duration of five sequential delete round trips.
        """
        keys_by_tier: dict[str, list[str]] = {}
        with self._lock:
            for tier in _TIERS:
                keys_by_tier[tier] = list(self._keys_by_tier[tier])
                self._keys_by_tier[tier].clear()
                self._stats[tier] = {"hits": 0, "misses": 0, "sets": 0}

        for keys in keys_by_tier.values():
            if keys:
                self._backend.delete(keys)


query_cache = QueryCacheService()
"""Process-wide cache. Import this, not the class."""
