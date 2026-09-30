"""Unit seam: pushing a finished eval run to Langfuse (dataset items, a run per
profile, and per-golden scores linked to the trace that produced them).
The Langfuse client is faked; nothing here touches the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from eval import langfuse_link


class _FakeRunItems:
    def __init__(self) -> None:
        self.created: list[Any] = []

    def create(self, *, request: Any) -> None:
        self.created.append(request)


class _FakeApi:
    def __init__(self) -> None:
        self.dataset_run_items = _FakeRunItems()


class _FakeClient:
    def __init__(self) -> None:
        self.api = _FakeApi()
        self.datasets: list[str] = []
        self.items: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.flushed = False

    def create_dataset(self, *, name: str, **_: Any) -> None:
        self.datasets.append(name)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs)

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": "q-001",
        "question": "What is a Pod? Mail bob@example.com",
        "ground_truth": "pod, container",
        "demonstrates_feature": "baseline",
        "intent": "rag",
        "trace_id": "t-1",
        "forbidden_check": {"passed": True},
        "route_check": None,
        "web_fallback_check": None,
        "source_overlap": {"overlap_pct": 1.0},
        "ragas_metrics": {"faithfulness": 0.9, "context_recall": float("nan")},
    }
    row.update(overrides)
    return row


def test_push_results_is_a_no_op_without_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langfuse_link, "get_client", lambda: None)
    assert langfuse_link.push_results([_row()], profile="all", run_stamp="x") is False


def test_push_results_creates_dataset_item_run_link_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(langfuse_link, "get_client", lambda: client)

    assert langfuse_link.push_results([_row()], profile="all", run_stamp="20260929") is True

    assert client.datasets == [langfuse_link.DATASET_NAME]
    (item,) = client.items
    assert item["dataset_name"] == langfuse_link.DATASET_NAME
    assert item["id"] == "q-001"
    assert item["input"] == {"question": "What is a Pod? Mail [EMAIL]"}  # redacted
    (link,) = client.api.dataset_run_items.created
    assert (link.run_name, link.dataset_item_id, link.trace_id) == ("all/20260929", "q-001", "t-1")

    by_name = {s["name"]: s for s in client.scores}
    assert by_name["faithfulness"]["value"] == 0.9
    assert "context_recall" not in by_name  # NaN is skipped, not sent
    assert by_name["golden_passed"]["value"] == 1
    assert all(s["trace_id"] == "t-1" for s in client.scores)
    assert client.flushed is True


def test_rows_without_a_trace_id_still_get_a_dataset_item_but_no_link_or_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(langfuse_link, "get_client", lambda: client)

    langfuse_link.push_results([_row(trace_id=None)], profile="all", run_stamp="x")

    assert len(client.items) == 1
    assert client.api.dataset_run_items.created == []
    assert client.scores == []


def test_a_langfuse_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()

    def boom(**_: Any) -> None:
        raise RuntimeError("langfuse down")

    client.create_dataset = boom  # type: ignore[method-assign]
    monkeypatch.setattr(langfuse_link, "get_client", lambda: client)

    assert langfuse_link.push_results([_row()], profile="all", run_stamp="x") is False


def test_invoke_traced_returns_the_invoker_result_and_no_trace_id_when_disabled() -> None:
    response = SimpleNamespace(answer="an answer")

    class _Invoker:
        def invoke(self, question: str, flags: dict, intent: str) -> tuple[Any, list]:
            return response, []

    result, trace_id = langfuse_link.invoke_traced(_Invoker(), "q?", {}, "rag")  # type: ignore[arg-type]
    assert result == (response, [])
    assert trace_id is None
