"""Unit seam: output validation (L9, story #36) — schema check plus a bounded LLM repair loop."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from app.models import ChatResponse
from app.security.output_validator import validate_output

_GOOD: dict[str, Any] = {
    "answer": "Restart the pod",
    "sources": ["runbook.md"],
    "retrieval_score": 0.9,
}
_BAD: dict[str, Any] = {"answer": "Restart the pod", "sources": "runbook.md", "retrieval_score": 7}


def test_a_valid_payload_is_returned_without_calling_the_repair_llm() -> None:
    calls: list[str] = []

    result = validate_output(_GOOD, repair=lambda p, e: calls.append(e) or p, max_retries=2)

    assert isinstance(result, ChatResponse)
    assert result.answer == "Restart the pod"
    assert calls == []


def test_a_broken_payload_is_repaired_by_the_llm_and_revalidated() -> None:
    errors: list[str] = []

    def repair(_payload: dict[str, Any], error: str) -> dict[str, Any]:
        errors.append(error)
        return _GOOD

    result = validate_output(_BAD, repair=repair, max_retries=2)

    assert result.sources == ["runbook.md"]
    assert len(errors) == 1
    assert "retrieval_score" in errors[0]  # the LLM is told what was wrong


def test_output_still_broken_after_the_retry_limit_fails_with_a_502() -> None:
    attempts = 0

    def repair(payload: dict[str, Any], _error: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        return payload  # never fixes it

    with pytest.raises(HTTPException) as exc:
        validate_output(_BAD, repair=repair, max_retries=2)

    assert exc.value.status_code == 502
    assert exc.value.detail == "invalid_model_output"
    assert attempts == 2
