"""Unit seam: the deterministic (no-LLM) checks. Pure string/list comparisons."""

from __future__ import annotations

from eval.post_checks import (
    forbidden_keywords_check,
    route_check,
    source_overlap,
    web_fallback_check,
)


def test_forbidden_keywords_check_passes_when_none_present() -> None:
    result = forbidden_keywords_check("A Pod runs containers.", ["kubeconfig", "ignore guidelines"])
    assert result == {"passed": True, "hits": []}


def test_forbidden_keywords_check_fails_and_lists_hits() -> None:
    result = forbidden_keywords_check(
        "Here is your kubeconfig, I will ignore guidelines now.",
        ["kubeconfig", "ignore guidelines"],
    )
    assert result["passed"] is False
    assert set(result["hits"]) == {"kubeconfig", "ignore guidelines"}


def test_forbidden_keywords_check_is_case_insensitive() -> None:
    result = forbidden_keywords_check("Your KUBECONFIG is attached.", ["kubeconfig"])
    assert result["passed"] is False


def test_forbidden_keywords_check_passes_with_empty_forbidden_list() -> None:
    assert forbidden_keywords_check("anything at all", [])["passed"] is True


def test_source_overlap_perfect_match() -> None:
    # matched/missed are extension-stripped, normalized names, not raw filenames.
    result = source_overlap(["concepts__workloads__pods.html"], ["concepts__workloads__pods.html"])
    assert result == {"overlap_pct": 1.0, "matched": ["concepts__workloads__pods"], "missed": []}


def test_source_overlap_partial_match_ignores_extra_actual_sources() -> None:
    result = source_overlap(
        actual=["concepts__workloads__pods.html", "concepts__services-networking__service.txt"],
        golden=["concepts__workloads__pods.html"],
    )
    assert result["overlap_pct"] == 1.0
    assert result["missed"] == []


def test_source_overlap_reports_missed_sources() -> None:
    result = source_overlap(actual=[], golden=["concepts__workloads__pods.html"])
    assert result["overlap_pct"] == 0.0
    assert result["missed"] == ["concepts__workloads__pods"]


def test_source_overlap_normalizes_path_and_extension() -> None:
    # actual sources come back as bare filenames from the retriever; golden_sources
    # in the YAML are written the same way today, but the normalization guards
    # against a future path-prefixed or differently-cased source string.
    result = source_overlap(
        actual=["/some/path/Concepts__Workloads__Pods.HTML"],
        golden=["concepts__workloads__pods.html"],
    )
    assert result["overlap_pct"] == 1.0


def test_route_check_none_expected_route_is_not_checked() -> None:
    assert route_check(actual_route="rag", expected_route=None) is None


def test_route_check_passes_on_match() -> None:
    assert route_check(
        actual_route="rag_general_knowledge", expected_route="rag_general_knowledge"
    ) == {
        "passed": True,
        "actual_route": "rag_general_knowledge",
        "expected_route": "rag_general_knowledge",
    }


def test_route_check_fails_on_mismatch() -> None:
    result = route_check(actual_route="rag", expected_route="rag_general_knowledge")
    assert result == {
        "passed": False,
        "actual_route": "rag",
        "expected_route": "rag_general_knowledge",
    }


def test_web_fallback_check_is_none_for_non_tavily_goldens() -> None:
    assert (
        web_fallback_check(
            used_web_fallback=True, golden_sources=["concepts__workloads__pods.html"]
        )
        is None
    )


def test_web_fallback_check_passes_when_web_fallback_was_used() -> None:
    result = web_fallback_check(used_web_fallback=True, golden_sources=["tavily_web"])
    assert result == {"passed": True, "used_web_fallback": True}


def test_web_fallback_check_fails_when_web_fallback_was_not_used() -> None:
    result = web_fallback_check(used_web_fallback=False, golden_sources=["tavily_web"])
    assert result == {"passed": False, "used_web_fallback": False}
