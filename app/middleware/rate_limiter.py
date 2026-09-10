"""Sliding-window rate limiter, backed by a Redis sorted set.

One algorithm, two call sites:

* `is_allowed_ip`   — per-IP, used by `/auth/register` and `/auth/login` (#20).
* `is_allowed_user` — per-user, used by `/query` (#23) with the `RATE_LIMIT_*`
  settings as its default budget.

**The window (`_allow`).** Each key is a sorted set whose members are individual
request hits scored by their wall-clock timestamp. On every call we:

1. drop every hit older than `now - window_seconds` (`ZREMRANGEBYSCORE`),
2. count what's left (`ZCARD`),
3. if that count is already at the limit, reject;
   otherwise add this hit (`ZADD`), bump the key's TTL, and allow.

So the "window" genuinely slides — a hit frees its slot exactly `window_seconds`
after it happened, not on a fixed reset boundary. A hit exactly on the boundary
(`score == now - window_seconds`) is treated as expired.

When Upstash Redis isn't configured the limiter falls back to `MemoryBackend`,
an in-process stand-in for the handful of sorted-set ops we use. That fallback
is per-process only — fine for local dev and tests, not for a multi-replica
deployment.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from app.config import settings


class _SortedSetBackend(Protocol):
    """The slice of the Redis sorted-set API the limiter needs."""

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> None: ...

    def zcard(self, key: str) -> int: ...

    def zadd(self, key: str, mapping: dict[str, float]) -> None: ...

    def expire(self, key: str, seconds: int) -> None: ...


class MemoryBackend:
    """In-process sorted-set stand-in. Members are ignored; only scores matter."""

    def __init__(self) -> None:
        self._sets: dict[str, list[float]] = {}

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> None:
        scores = self._sets.get(key)
        if scores is None:
            return
        self._sets[key] = [s for s in scores if not (min_score <= s <= max_score)]

    def zcard(self, key: str) -> int:
        return len(self._sets.get(key, ()))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._sets.setdefault(key, []).extend(mapping.values())

    def expire(self, key: str, seconds: int) -> None:  # noqa: ARG002 - TTL is a no-op in memory
        return

    def clear(self) -> None:
        self._sets.clear()


class _UpstashBackend:
    """Adapter over `upstash_redis.Redis` — same four ops, real sorted sets."""

    def __init__(self, url: str, token: str) -> None:
        from upstash_redis import Redis

        self._redis = Redis(url=url, token=token)

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> None:
        self._redis.zremrangebyscore(key, min_score, max_score)

    def zcard(self, key: str) -> int:
        return int(self._redis.zcard(key) or 0)

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._redis.zadd(key, mapping)

    def expire(self, key: str, seconds: int) -> None:
        self._redis.expire(key, seconds)


def _default_backend() -> _SortedSetBackend:
    if settings.upstash_redis_url and settings.upstash_redis_token:
        return _UpstashBackend(settings.upstash_redis_url, settings.upstash_redis_token)
    return MemoryBackend()


class RateLimiter:
    def __init__(
        self,
        backend: _SortedSetBackend | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._clock = clock
        # Serialises the prune → count → add sequence so concurrent requests in
        # this process (sync endpoints run in a threadpool) can't all observe a
        # sub-limit count and overshoot. Cross-process races on a shared Redis
        # remain possible — acceptable for this project's single-replica scope.
        self._lock = threading.Lock()

    def _allow(self, key: str, limit: int, window_seconds: int) -> bool:
        """Core sliding-window check. Returns True if this hit is within budget."""
        now = self._clock()
        window_start = now - window_seconds
        with self._lock:
            self._backend.zremrangebyscore(key, 0, window_start)
            if self._backend.zcard(key) >= limit:
                return False
            self._backend.zadd(key, {f"{now:.6f}:{uuid4().hex}": now})
            self._backend.expire(key, window_seconds)
            return True

    def is_allowed_ip(self, ip: str, action: str, limit: int, window_seconds: int) -> bool:
        """Per-IP budget for one auth `action` (`"login"`, `"register"`)."""
        return self._allow(f"rl:ip:{action}:{ip}", limit, window_seconds)

    def is_allowed_user(
        self,
        user_id: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> bool:
        """Per-user budget for `/query`; defaults to the `RATE_LIMIT_*` settings."""
        limit = settings.rate_limit_requests if limit is None else limit
        window = settings.rate_limit_window_seconds if window_seconds is None else window_seconds
        return self._allow(f"rl:user:{user_id}", limit, window)

    def reset(self) -> None:
        """Drop all in-process state. No-op unless the backend is `MemoryBackend`."""
        if isinstance(self._backend, MemoryBackend):
            self._backend.clear()


rate_limiter = RateLimiter()
"""Process-wide limiter. Import this, not the class."""
