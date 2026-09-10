"""Unit seam: the sliding-window rate-limiter math.

The window is driven by an injected clock so these tests are deterministic and
never sleep. Backend is the in-memory sorted-set stand-in (no Redis needed).
"""

import pytest

from app.middleware.rate_limiter import MemoryBackend, RateLimiter


@pytest.fixture
def clock() -> list[float]:
    return [1_000.0]


@pytest.fixture
def limiter(clock: list[float]) -> RateLimiter:
    return RateLimiter(backend=MemoryBackend(), clock=lambda: clock[0])


def test_allows_exactly_the_limit_then_blocks(limiter: RateLimiter) -> None:
    for _ in range(5):
        assert limiter._allow("k", limit=5, window_seconds=60) is True
    assert limiter._allow("k", limit=5, window_seconds=60) is False


def test_window_fully_slides_past_unblocks(limiter: RateLimiter, clock: list[float]) -> None:
    for _ in range(5):
        limiter._allow("k", limit=5, window_seconds=60)
    assert limiter._allow("k", limit=5, window_seconds=60) is False

    clock[0] += 61  # every earlier hit is now outside the window
    assert limiter._allow("k", limit=5, window_seconds=60) is True


def test_window_slides_partially_frees_one_slot(limiter: RateLimiter, clock: list[float]) -> None:
    for i in range(5):
        clock[0] = 1_000.0 + i
        assert limiter._allow("k", limit=5, window_seconds=60) is True

    clock[0] = 1_000.0 + 60  # only the first hit (score 1000) ages out
    assert limiter._allow("k", limit=5, window_seconds=60) is True
    assert limiter._allow("k", limit=5, window_seconds=60) is False


def test_keys_are_isolated(limiter: RateLimiter) -> None:
    for _ in range(5):
        assert limiter._allow("a", limit=5, window_seconds=60) is True
    assert limiter._allow("a", limit=5, window_seconds=60) is False
    assert limiter._allow("b", limit=5, window_seconds=60) is True


def test_is_allowed_ip_scopes_by_action_and_ip(limiter: RateLimiter) -> None:
    assert limiter.is_allowed_ip("1.2.3.4", "login", limit=2, window_seconds=60) is True
    assert limiter.is_allowed_ip("1.2.3.4", "login", limit=2, window_seconds=60) is True
    assert limiter.is_allowed_ip("1.2.3.4", "login", limit=2, window_seconds=60) is False
    # a different action for the same IP has its own budget
    assert limiter.is_allowed_ip("1.2.3.4", "register", limit=2, window_seconds=60) is True
    # a different IP has its own budget
    assert limiter.is_allowed_ip("9.9.9.9", "login", limit=2, window_seconds=60) is True


def test_is_allowed_user_uses_settings_defaults(limiter: RateLimiter) -> None:
    from app.config import settings

    for _ in range(settings.rate_limit_requests):
        assert limiter.is_allowed_user("user-1") is True
    assert limiter.is_allowed_user("user-1") is False
    assert limiter.is_allowed_user("user-2") is True


def test_memory_backend_drops_keys_that_age_out(clock: list[float]) -> None:
    backend = MemoryBackend()
    limiter = RateLimiter(backend=backend, clock=lambda: clock[0])
    limiter._allow("k", limit=5, window_seconds=60)
    assert "k" in backend._sets

    clock[0] += 61  # the only hit ages out on the next touch
    limiter._allow("k", limit=5, window_seconds=60)
    # the pre-add prune emptied the set and removed the key; then this hit re-added it
    assert backend.zcard("k") == 1


def test_reset_clears_memory_state(limiter: RateLimiter) -> None:
    for _ in range(5):
        limiter._allow("k", limit=5, window_seconds=60)
    assert limiter._allow("k", limit=5, window_seconds=60) is False
    limiter.reset()
    assert limiter._allow("k", limit=5, window_seconds=60) is True
