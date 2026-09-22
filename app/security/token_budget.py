"""Per-user daily token budget (L6, story #30).

Two moments in a request:

* `check`   — early: is this user already at or over today's allowance? If so,
  refuse with a 429 before any LLM money is spent.
* `consume` — last: add the tokens the finished request used. We only know the
  cost at the end, and a failed request should not be charged.

A user can therefore overshoot the limit on their final request; the next
`check` then refuses them. The counter is keyed by user and UTC day, so a new
day starts at zero with no reset job.

Storage is a plain Redis counter (`INCRBY` + `EXPIRE`). With no Upstash
configured it falls back to an in-process dict — per-process only, fine for
local dev and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from fastapi import HTTPException, status

from app.config import settings

_KEY_TTL_SECONDS = 2 * 24 * 3600  # outlives the day it counts, then cleans itself up


class _CounterBackend(Protocol):
    def get(self, key: str) -> int: ...

    def incrby(self, key: str, amount: int) -> None: ...

    def expire(self, key: str, seconds: int) -> None: ...


class MemoryBudgetBackend:
    """In-process counter stand-in. TTL is a no-op."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def get(self, key: str) -> int:
        return self._counts.get(key, 0)

    def incrby(self, key: str, amount: int) -> None:
        self._counts[key] = self._counts.get(key, 0) + amount

    def expire(self, key: str, seconds: int) -> None:  # noqa: ARG002
        return

    def clear(self) -> None:
        self._counts.clear()


class _UpstashBudgetBackend:
    def __init__(self, url: str, token: str) -> None:
        from upstash_redis import Redis

        self._redis = Redis(url=url, token=token)

    def get(self, key: str) -> int:
        return int(self._redis.get(key) or 0)

    def incrby(self, key: str, amount: int) -> None:
        self._redis.incrby(key, amount)

    def expire(self, key: str, seconds: int) -> None:
        self._redis.expire(key, seconds)


def _default_backend() -> _CounterBackend:
    if settings.upstash_redis_url and settings.upstash_redis_token:
        return _UpstashBudgetBackend(settings.upstash_redis_url, settings.upstash_redis_token)
    return MemoryBudgetBackend()


class TokenBudget:
    def __init__(
        self,
        backend: _CounterBackend | None = None,
        daily_limit: int | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._daily_limit = daily_limit
        self._clock = clock

    def _key(self, user_id: int) -> str:
        return f"budget:{user_id}:{self._clock().date().isoformat()}"

    @property
    def daily_limit(self) -> int:
        return (
            settings.max_tokens_per_user_daily if self._daily_limit is None else self._daily_limit
        )

    def check(self, user_id: int) -> None:
        if self._backend.get(self._key(user_id)) >= self.daily_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="token_budget_exceeded"
            )

    def consume(self, user_id: int, tokens: int) -> None:
        key = self._key(user_id)
        self._backend.incrby(key, tokens)
        self._backend.expire(key, _KEY_TTL_SECONDS)

    def reset(self) -> None:
        """Drop all in-process state. No-op unless the backend is `MemoryBudgetBackend`."""
        if isinstance(self._backend, MemoryBudgetBackend):
            self._backend.clear()


token_budget = TokenBudget()
"""Process-wide budget. Import this, not the class."""
