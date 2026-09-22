"""Unit seam: the per-user daily token budget (L6).

`check` refuses a user who is already at or over their allowance; `consume`
adds the tokens a finished request actually used. The counter is keyed by
user and UTC day, so a new day starts fresh.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.security.token_budget import MemoryBudgetBackend, TokenBudget


def _budget(limit: int = 1000, now: datetime | None = None) -> TokenBudget:
    moment = now or datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    return TokenBudget(MemoryBudgetBackend(), daily_limit=limit, clock=lambda: moment)


def test_a_user_under_budget_passes_the_check() -> None:
    budget = _budget(limit=1000)
    budget.consume(user_id=1, tokens=950)

    budget.check(user_id=1)  # 950 < 1000, no error


def test_a_user_at_or_over_budget_gets_a_429() -> None:
    budget = _budget(limit=1000)
    budget.consume(user_id=1, tokens=950)
    budget.consume(user_id=1, tokens=120)  # 1070, a request may overshoot on its last call

    with pytest.raises(HTTPException) as exc:
        budget.check(user_id=1)

    assert exc.value.status_code == 429
    assert exc.value.detail == "token_budget_exceeded"


def test_budgets_are_per_user() -> None:
    budget = _budget(limit=100)
    budget.consume(user_id=1, tokens=500)

    budget.check(user_id=2)  # a different user is unaffected


def test_a_new_day_starts_fresh() -> None:
    backend = MemoryBudgetBackend()
    today = datetime(2026, 9, 21, 23, 59, tzinfo=UTC)
    tomorrow = datetime(2026, 9, 22, 0, 1, tzinfo=UTC)
    TokenBudget(backend, daily_limit=100, clock=lambda: today).consume(user_id=1, tokens=500)

    TokenBudget(backend, daily_limit=100, clock=lambda: tomorrow).check(user_id=1)
