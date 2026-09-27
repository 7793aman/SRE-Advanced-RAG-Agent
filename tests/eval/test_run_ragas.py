"""Unit seam: the CLI's own orchestration logic — row-building, skip handling,
and the empty-context guard around Ragas — with the invoker and Ragas faked.
Not a test that the real pipeline or Ragas work; that's an integration concern
outside pytest's reach (needs OpenAI + a running app).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from app.models import ChatResponse, ResponseMetadata
from eval.invokers import SkippedIntent
from eval.run_ragas import build_row, default_output_path, main, run_eval
from eval.schema import Golden

_GOLDEN = Golden(
    id="q-001",
    question="What is a Pod?",
    intent="rag",
    golden_sources=["concepts__workloads__pods.html"],
    golden_answer_keywords=["pod", "container"],
    demonstrates_feature="baseline",
    expected_baseline="pass",
    expected_with_feature="pass",
    notes="control case",
)

_ADAPTIVE_GOLDEN = Golden(
    id="q-041",
    question="What temperature does water boil at?",
    intent="rag",
    golden_sources=["no_retrieval_expected"],
    golden_answer_keywords=["100", "celsius"],
    demonstrates_feature="adaptive_retrieval",
    expected_baseline="fail",
    expected_with_feature="pass",
    expected_route="rag_general_knowledge",
    notes="skip-retrieval case",
)

_CRAG_GOLDEN = Golden(
    id="q-017",
    question="What's the weather forecast for Seattle this weekend?",
    intent="web_fallback",
    golden_sources=["tavily_web"],
    golden_answer_keywords=["weather", "Seattle"],
    demonstrates_feature="crag",
    expected_baseline="fail",
    expected_with_feature="pass",
    notes="web fallback case",
)


def _response(
    route: str = "rag",
    answer: str = "A Pod is a thing.",
    sources: list[str] | None = None,
    used_web_fallback: bool = False,
) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=sources if sources is not None else ["concepts__workloads__pods.html"],
        retrieval_score=0.9,
        metadata=ResponseMetadata(route=route, used_web_fallback=used_web_fallback),
    )


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


def test_default_output_path_is_unique_to_the_microsecond_not_just_the_second() -> None:
    t1 = datetime.datetime(2026, 1, 1, 12, 0, 0, 100000, tzinfo=datetime.UTC)
    t2 = datetime.datetime(2026, 1, 1, 12, 0, 0, 200000, tzinfo=datetime.UTC)
    p1 = default_output_path(t1, "naive")
    p2 = default_output_path(t2, "naive")
    assert p1 != p2
    # Still lexically sorts newest-last, since find_latest() just sorts filenames.
    assert sorted([str(p2), str(p1)]) == [str(p1), str(p2)]


def test_build_row_shapes_the_fields_ragas_and_post_checks_need() -> None:
    row = build_row(_GOLDEN, _response(), [_FakeChunk("A Pod is the smallest deployable unit.")])
    assert row["id"] == "q-001"
    assert row["answer"] == "A Pod is a thing."
    assert row["contexts"] == ["A Pod is the smallest deployable unit."]
    assert row["ground_truth"] == "pod, container"
    assert row["actual_sources"] == ["concepts__workloads__pods.html"]
    assert row["golden_sources"] == ["concepts__workloads__pods.html"]
    assert row["expected_baseline"] == "pass"
    assert row["expected_with_feature"] == "pass"


def test_build_row_computes_source_overlap_and_forbidden_check_immediately() -> None:
    row = build_row(_GOLDEN, _response(), [_FakeChunk("...")])
    assert row["source_overlap"]["overlap_pct"] == 1.0
    assert row["forbidden_check"]["passed"] is True


def test_build_row_computes_route_check_only_when_golden_has_an_expected_route() -> None:
    plain = build_row(_GOLDEN, _response(route="rag"), [])
    assert plain["route_check"] is None

    adaptive = build_row(_ADAPTIVE_GOLDEN, _response(route="rag_general_knowledge", sources=[]), [])
    assert adaptive["route_check"] == {
        "passed": True,
        "actual_route": "rag_general_knowledge",
        "expected_route": "rag_general_knowledge",
    }


def test_build_row_uses_web_fallback_check_for_tavily_web_goldens_not_source_overlap() -> None:
    row = build_row(
        _CRAG_GOLDEN,
        _response(sources=["some-real-tavily-result.com"], used_web_fallback=True),
        [],
    )
    # source_overlap is 0 here (a real URL never matches the "tavily_web" sentinel) —
    # web_fallback_check is what actually gates this golden.
    assert row["source_overlap"]["overlap_pct"] == 0.0
    assert row["web_fallback_check"] == {"passed": True, "used_web_fallback": True}


def test_run_eval_skips_unsupported_intents_instead_of_raising() -> None:
    class _SqlSkippingInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            raise SkippedIntent(f"intent={intent} not supported in service mode")

    sql_golden = Golden(
        id="q-025",
        question="How many pods?",
        intent="sql",
        golden_sources=["query_results"],
        golden_answer_keywords=["pod"],
        demonstrates_feature="sql",
        expected_baseline="pass",
        expected_with_feature="pass",
        notes="sql case",
    )
    rows, skipped = run_eval([sql_golden], flags={}, invoker=_SqlSkippingInvoker())
    assert rows == []
    assert skipped == [{"id": "q-025", "reason": "intent=sql not supported in service mode"}]


def test_run_eval_records_unexpected_errors_as_skips_not_crashes() -> None:
    class _BoomInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            raise RuntimeError("openai timeout")

    rows, skipped = run_eval([_GOLDEN], flags={}, invoker=_BoomInvoker())
    assert rows == []
    assert skipped == [{"id": "q-001", "reason": "error: openai timeout"}]


def test_run_eval_builds_one_row_per_successful_invocation() -> None:
    class _FakeInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            return _response(), [_FakeChunk("chunk text")]

    rows, skipped = run_eval([_GOLDEN], flags={}, invoker=_FakeInvoker())
    assert skipped == []
    assert len(rows) == 1
    assert rows[0]["id"] == "q-001"


def test_main_writes_a_results_json_with_the_expected_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions_path = tmp_path / "goldens.yaml"
    questions_path.write_text(
        "- id: q-001\n"
        '  question: "What is a Pod?"\n'
        "  intent: rag\n"
        "  golden_sources: [concepts__workloads__pods.html]\n"
        "  golden_answer_keywords: [pod, container]\n"
        "  demonstrates_feature: baseline\n"
        "  expected_baseline: pass\n"
        "  expected_with_feature: pass\n"
        "  notes: control case\n"
    )
    output_path = tmp_path / "result.json"

    class _FakeInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            return _response(), [_FakeChunk("chunk text")]

    monkeypatch.setattr("eval.run_ragas.ServiceInvoker", lambda: _FakeInvoker())
    monkeypatch.setattr(
        "eval.run_ragas.run_ragas_metrics",
        lambda rows: [
            {
                "faithfulness": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "answer_relevancy": 1.0,
            }
            for _ in rows
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ragas",
            "--profile",
            "naive",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )
    main()

    payload = json.loads(output_path.read_text())
    assert payload["profile"] == "naive"
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["ragas_metrics"]["faithfulness"] == 1.0
    assert payload["aggregate"]["passed"] == 1
    assert payload["expected_outcome_mismatches"] == []


def test_main_flags_a_config_skip_as_unverified_but_not_an_intent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    questions_path = tmp_path / "goldens.yaml"
    questions_path.write_text(
        "- id: q-017\n"
        '  question: "What is the weather?"\n'
        "  intent: web_fallback\n"
        "  golden_sources: [tavily_web]\n"
        "  golden_answer_keywords: [weather]\n"
        "  demonstrates_feature: crag\n"
        "  expected_baseline: fail\n"
        "  expected_with_feature: pass\n"
        "  notes: crag case\n"
        "- id: q-025\n"
        '  question: "How many pods?"\n'
        "  intent: sql\n"
        "  golden_sources: [query_results]\n"
        "  golden_answer_keywords: [pod]\n"
        "  demonstrates_feature: sql\n"
        "  expected_baseline: pass\n"
        "  expected_with_feature: pass\n"
        "  notes: sql case\n"
    )
    output_path = tmp_path / "result.json"

    class _AllSkippingInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            if intent == "sql":
                raise SkippedIntent("intent=sql not supported in service mode")
            raise SkippedIntent("tavily_unset: TAVILY_API_KEY not configured")

    monkeypatch.setattr("eval.run_ragas.ServiceInvoker", lambda: _AllSkippingInvoker())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ragas",
            "--profile",
            "naive",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    payload = json.loads(output_path.read_text())
    assert [s["id"] for s in payload["skipped"]] == ["q-017", "q-025"]
    # The sql skip is the accepted, permanent limitation — not "unverified".
    # The tavily skip is a config problem — a real "this wasn't checked" gap.
    assert [s["id"] for s in payload["unverified"]] == ["q-017"]
    assert "q-017" in capsys.readouterr().err


def test_main_warns_loudly_on_stderr_when_a_golden_errors_not_just_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    questions_path = tmp_path / "goldens.yaml"
    questions_path.write_text(
        "- id: q-001\n"
        '  question: "What is a Pod?"\n'
        "  intent: rag\n"
        "  golden_sources: [concepts__workloads__pods.html]\n"
        "  golden_answer_keywords: [pod, container]\n"
        "  demonstrates_feature: baseline\n"
        "  expected_baseline: pass\n"
        "  expected_with_feature: pass\n"
        "  notes: control case\n"
    )
    output_path = tmp_path / "result.json"

    class _BoomInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            raise RuntimeError("openai timeout")

    monkeypatch.setattr("eval.run_ragas.ServiceInvoker", lambda: _BoomInvoker())
    monkeypatch.setattr("eval.run_ragas.run_ragas_metrics", lambda rows: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ragas",
            "--profile",
            "naive",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "q-001" in err
    assert "openai timeout" in err


def test_main_does_not_crash_when_ragas_returns_fewer_records_than_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    questions_path = tmp_path / "goldens.yaml"
    questions_path.write_text(
        "- id: q-001\n"
        '  question: "What is a Pod?"\n'
        "  intent: rag\n"
        "  golden_sources: [concepts__workloads__pods.html]\n"
        "  golden_answer_keywords: [pod, container]\n"
        "  demonstrates_feature: baseline\n"
        "  expected_baseline: pass\n"
        "  expected_with_feature: pass\n"
        "  notes: control case\n"
        "- id: q-002\n"
        '  question: "What is a Deployment?"\n'
        "  intent: rag\n"
        "  golden_sources: [concepts__workloads__controllers__deployment.txt]\n"
        "  golden_answer_keywords: [deployment]\n"
        "  demonstrates_feature: baseline\n"
        "  expected_baseline: pass\n"
        "  expected_with_feature: pass\n"
        "  notes: control case\n"
    )
    output_path = tmp_path / "result.json"

    class _FakeInvoker:
        def invoke(self, question: str, flags: dict, intent: str) -> Any:
            return _response(), [_FakeChunk("chunk text")]

    monkeypatch.setattr("eval.run_ragas.ServiceInvoker", lambda: _FakeInvoker())
    # Simulates a judge-LLM error on one sample: 2 rows in, only 1 record back.
    monkeypatch.setattr(
        "eval.run_ragas.run_ragas_metrics",
        lambda rows: [
            {
                "faithfulness": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "answer_relevancy": 1.0,
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ragas",
            "--profile",
            "naive",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    main()  # must not raise

    payload = json.loads(output_path.read_text())
    assert len(payload["rows"]) == 2
    scored = [r for r in payload["rows"] if r["ragas_metrics"] is not None]
    unscored = [r for r in payload["rows"] if r["ragas_metrics"] is None]
    assert len(scored) == 1
    assert len(unscored) == 1
    assert "WARNING" in capsys.readouterr().err
