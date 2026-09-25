"""Thread-safe lazy singleton: builds a value once, on first use, and caches
it for the life of the process.

`functools.lru_cache`/`@cache` on a zero-arg builder look like they do this,
but don't lock — two requests racing into a cold cache both call the builder
concurrently. That's merely wasteful for a cheap client constructor, but
`content_guard.py`'s ML scanners (loaded onto Apple's MPS backend) actually
deadlocked the whole process under that exact race (found testing issue
#34's demo UI). This factors that fix out so every lazy-built client/model
in the app can use it instead of `lru_cache`.

A build that raises is not cached, so the next call retries — a transient
failure (a network blip on first use) shouldn't pin the process to a broken
state for its whole lifetime. Same contract `lru_cache` already gave callers
(it doesn't cache exceptions either), so switching to this is a drop-in
replacement.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class LazySingleton[T]:
    def __init__(self, build: Callable[[], T]) -> None:
        self._build = build
        self._value: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        if self._value is not None:
            return self._value
        with self._lock:
            if self._value is None:
                self._value = self._build()
        return self._value

    def reset(self) -> None:
        """Test-only escape hatch — the `lru_cache.cache_clear()` this
        singleton replaced offered the same thing, for tests that need a
        fresh instance between cases (e.g. a real Postgres-backed graph)."""
        with self._lock:
            self._value = None
