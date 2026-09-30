"""Unit seam: the tracing wrapper is a pass-through no-op while Langfuse is
unconfigured, and redacts PII from payloads before they can leave the process.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services import tracing


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")


def test_disabled_without_keys() -> None:
    assert tracing.is_enabled() is False


def test_enabled_needs_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-x")
    assert tracing.is_enabled() is False
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-x")
    assert tracing.is_enabled() is True


def test_observe_is_a_pass_through_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("must not build a Langfuse client while disabled")

    monkeypatch.setattr(tracing, "_get_client", boom)

    @tracing.observe(name="add")
    def add(a: int, b: int = 0) -> int:
        return a + b

    assert add(2, b=3) == 5
    assert add.__name__ == "add"


def test_observe_propagates_exceptions_when_disabled() -> None:
    @tracing.observe()
    def fail() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        fail()


def test_helpers_are_no_ops_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_get_client", lambda: (_ for _ in ()).throw(AssertionError))
    tracing.update_generation(model="m", usage={"input": 1, "output": 2})
    tracing.update_trace(metadata={"flags": {}})
    assert tracing.callbacks() == []
    assert tracing.current_trace_id() is None
    tracing.flush()


def test_mask_redacts_pii_in_nested_payloads() -> None:
    payload = {
        "question": "mail bob@example.com from 10.0.0.1",
        "chunks": [{"text": "call +1 415 555 2671"}, "plain"],
        "score": 0.5,
    }
    assert tracing.mask(payload) == {
        "question": "mail [EMAIL] from [IP]",
        "chunks": [{"text": "call [PHONE]"}, "plain"],
        "score": 0.5,
    }


def test_mask_never_raises() -> None:
    class Odd:
        pass

    odd = Odd()
    assert tracing.mask(odd) is odd


def test_mask_covers_sets() -> None:
    assert tracing.mask({"bob@example.com"}) == ["[EMAIL]"]


def test_a_failed_client_build_is_not_retried_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    monkeypatch.setattr(tracing, "_build_failed_at", None)
    attempts: list[int] = []

    def fail() -> None:
        attempts.append(1)
        raise RuntimeError("bad host")

    monkeypatch.setattr(tracing, "_get_client", fail)
    assert tracing.get_client() is None
    assert tracing.get_client() is None
    assert len(attempts) == 1
    monkeypatch.setattr(tracing, "_build_failed_at", None)
