"""Startup warm-up: best-effort, and switchable off."""

import pytest

from app import warmup
from app.config import settings


def test_run_warmup_runs_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    steps = tuple((name, lambda n=name: ran.append(n)) for name in ("a", "b", "c"))
    monkeypatch.setattr(warmup, "_STEPS", steps)

    warmup.run_warmup()

    assert ran == ["a", "b", "c"]


def test_a_failing_step_does_not_stop_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    def boom() -> None:
        raise RuntimeError("model download failed")

    monkeypatch.setattr(warmup, "_STEPS", (("bad", boom), ("good", lambda: ran.append("good"))))

    warmup.run_warmup()

    assert ran == ["good"]


def test_start_warmup_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "warmup_on_startup", False)
    started: list[bool] = []
    monkeypatch.setattr(warmup.threading, "Thread", lambda *a, **k: started.append(True))

    warmup.start_warmup()

    assert started == []
