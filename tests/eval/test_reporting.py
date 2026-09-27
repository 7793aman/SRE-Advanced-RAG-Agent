"""Unit seam: turning per-row checks into a single PASS/FAIL, and diffing that
against what a golden's YAML said to expect. This is the piece issue #33 needs
that the reference tutorial never built (grep confirms `expected_baseline` /
`expected_with_feature` are validated on the YAML but never read downstream there).
Deliberately deterministic — no Ragas score is part of the gate; see the module
docstring in eval/reporting.py.
"""

from __future__ import annotations

from eval.reporting import aggregate, expected_outcome_mismatches, golden_passed

_BASE_ROW = {
    "id": "q-001",
    "expected_baseline": "pass",
    "expected_with_feature": "pass",
    "forbidden_check": {"passed": True, "hits": []},
    "source_overlap": {"overlap_pct": 1.0, "matched": ["pods"], "missed": []},
    "route_check": None,
    "ragas_metrics": {
        "faithfulness": 0.9,
        "context_precision": 0.8,
        "context_recall": 1.0,
        "answer_relevancy": 0.9,
    },
}


def test_golden_passed_true_when_forbidden_ok_and_all_sources_found() -> None:
    assert golden_passed(_BASE_ROW) is True


def test_golden_passed_false_when_forbidden_keyword_leaked() -> None:
    row = {**_BASE_ROW, "forbidden_check": {"passed": False, "hits": ["kubeconfig"]}}
    assert golden_passed(row) is False


def test_golden_passed_false_when_a_required_source_was_missed() -> None:
    row = {
        **_BASE_ROW,
        "source_overlap": {"overlap_pct": 0.5, "matched": ["pods"], "missed": ["x"]},
    }
    assert golden_passed(row) is False


def test_golden_passed_uses_route_check_instead_of_source_overlap_when_present() -> None:
    # adaptive_retrieval goldens: source_overlap is meaningless when retrieval was
    # correctly skipped, so a passing route_check overrides a "bad-looking" overlap.
    row = {
        **_BASE_ROW,
        "source_overlap": {"overlap_pct": 0.0, "matched": [], "missed": ["no_retrieval_expected"]},
        "route_check": {
            "passed": True,
            "actual_route": "rag_general_knowledge",
            "expected_route": "rag_general_knowledge",
        },
    }
    assert golden_passed(row) is True


def test_golden_passed_uses_web_fallback_check_instead_of_source_overlap_when_present() -> None:
    # CRAG goldens: golden_sources=[tavily_web] can never appear in actual_sources,
    # so a "bad-looking" overlap must not veto a correctly-used web fallback.
    row = {
        **_BASE_ROW,
        "source_overlap": {"overlap_pct": 0.0, "matched": [], "missed": ["tavily_web"]},
        "web_fallback_check": {"passed": True, "used_web_fallback": True},
    }
    assert golden_passed(row) is True


def test_golden_passed_false_when_web_fallback_check_present_and_failing() -> None:
    row = {**_BASE_ROW, "web_fallback_check": {"passed": False, "used_web_fallback": False}}
    assert golden_passed(row) is False


def test_golden_passed_false_when_route_check_present_and_failing() -> None:
    row = {
        **_BASE_ROW,
        "route_check": {
            "passed": False,
            "actual_route": "rag",
            "expected_route": "rag_general_knowledge",
        },
    }
    assert golden_passed(row) is False


def test_golden_passed_does_not_depend_on_ragas_scores() -> None:
    # Ragas metrics are reported alongside, not gated on — see module docstring.
    row = {
        **_BASE_ROW,
        "ragas_metrics": {
            "faithfulness": 0.01,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_relevancy": 0.0,
        },
    }
    assert golden_passed(row) is True


def test_expected_outcome_mismatches_empty_when_naive_run_matches_expected_baseline() -> None:
    rows = [_BASE_ROW]  # expected_baseline: pass, and it actually passes
    assert expected_outcome_mismatches(rows, profile="naive") == []


def test_expected_outcome_mismatches_flags_an_unexpected_pass_on_naive() -> None:
    # q-005-style golden: naive is EXPECTED to fail this, but it actually passed.
    row = {**_BASE_ROW, "id": "q-005", "expected_baseline": "fail"}
    mismatches = expected_outcome_mismatches([row], profile="naive")
    assert mismatches == [{"id": "q-005", "expected": "fail", "actual": "pass"}]


def test_expected_outcome_mismatches_flags_an_unexpected_fail_on_all() -> None:
    row = {**_BASE_ROW, "forbidden_check": {"passed": False, "hits": ["x"]}}
    mismatches = expected_outcome_mismatches([row], profile="all")
    assert mismatches == [{"id": "q-001", "expected": "pass", "actual": "fail"}]


def test_expected_outcome_mismatches_uses_expected_with_feature_for_non_naive_profiles() -> None:
    row = {**_BASE_ROW, "expected_with_feature": "pass"}
    assert expected_outcome_mismatches([row], profile="hybrid+rerank") == []


def test_aggregate_averages_ragas_metrics_and_counts_pass_fail() -> None:
    rows = [
        _BASE_ROW,
        {
            **_BASE_ROW,
            "id": "q-002",
            "ragas_metrics": {
                "faithfulness": 0.5,
                "context_precision": 0.4,
                "context_recall": 0.5,
                "answer_relevancy": 0.5,
            },
        },
    ]
    agg = aggregate(rows)
    assert agg["faithfulness"] == 0.7
    assert agg["forbidden_violations"] == 0
    assert agg["passed"] == 2
    assert agg["failed"] == 0


def test_aggregate_handles_no_rows() -> None:
    agg = aggregate([])
    assert agg["faithfulness"] is None
    assert agg["passed"] == 0
    assert agg["failed"] == 0
