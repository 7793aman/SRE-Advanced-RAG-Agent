"""Unit seam: the Ragas wiring itself (which fields go where, empty-input
short-circuit). `ragas.evaluate` is faked — this is not a test that Ragas's
metrics are correct, only that we hand it the right shape and get a list back.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from eval.ragas_adapter import _get_ragas_llm, build_dataset, run

_ROWS = [
    {
        "question": "What is a Pod?",
        "answer": "A Pod is the smallest deployable unit.",
        "contexts": ["A Pod is the smallest deployable unit in Kubernetes."],
        "ground_truth": "pod, container, namespace",
    }
]


def test_get_ragas_llm_overrides_a_stale_shell_env_var_not_just_setdefault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "stale-key-from-some-other-shell")
    monkeypatch.setattr("eval.ragas_adapter.settings.openai_api_key", "this-projects-real-key")
    monkeypatch.setattr("eval.ragas_adapter.llm_factory", lambda model: model)

    _get_ragas_llm()

    assert os.environ["OPENAI_API_KEY"] == "this-projects-real-key"


def test_build_dataset_maps_row_fields_to_ragas_field_names() -> None:
    ds = build_dataset(_ROWS)
    assert ds["user_input"] == ["What is a Pod?"]
    assert ds["response"] == ["A Pod is the smallest deployable unit."]
    assert ds["retrieved_contexts"] == [["A Pod is the smallest deployable unit in Kubernetes."]]
    assert ds["reference"] == ["pod, container, namespace"]


def test_run_returns_empty_list_for_no_rows_without_calling_ragas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("evaluate() should not be called for an empty row list")

    monkeypatch.setattr("eval.ragas_adapter.evaluate", _boom)
    assert run([]) == []


def test_run_calls_evaluate_with_the_four_metrics_and_returns_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeResult:
        def to_pandas(self) -> Any:
            class _Frame:
                def to_dict(self, orient: str) -> list[dict]:
                    assert orient == "records"
                    return [{"faithfulness": 1.0}]

            return _Frame()

    def _fake_evaluate(
        dataset: Any, metrics: Any, llm: Any, embeddings: Any, show_progress: bool
    ) -> Any:
        captured["metric_names"] = [m.name for m in metrics]
        captured["show_progress"] = show_progress
        return _FakeResult()

    monkeypatch.setattr("eval.ragas_adapter.evaluate", _fake_evaluate)
    monkeypatch.setattr("eval.ragas_adapter._get_ragas_llm", lambda: "fake-llm")
    monkeypatch.setattr("eval.ragas_adapter._get_ragas_embeddings", lambda: "fake-embeddings")

    result = run(_ROWS)

    assert result == [{"faithfulness": 1.0}]
    assert captured["metric_names"] == [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    ]
    assert captured["show_progress"] is False
