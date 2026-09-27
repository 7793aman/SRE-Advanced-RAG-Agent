"""Deterministic (no-LLM) checks — string/list comparisons, same result every time.

These catch what an LLM judge doesn't need to: did retrieval find the right doc at
all, did the answer leak something it shouldn't, did the route decision (adaptive
retrieval) go the way it should. See eval/ragas_adapter.py for the LLM-judge side.
"""

from __future__ import annotations

import os


def forbidden_keywords_check(answer: str, forbidden: list[str]) -> dict:
    answer_lower = answer.lower()
    hits = [kw for kw in forbidden if kw.lower() in answer_lower]
    return {"passed": not hits, "hits": hits}


def source_overlap(actual: list[str], golden: list[str]) -> dict:
    def _norm(s: str) -> str:
        return os.path.splitext(os.path.basename(s.strip().lower()))[0]

    actual_set = {_norm(s) for s in actual}
    golden_set = {_norm(s) for s in golden}
    overlap = actual_set & golden_set

    return {
        "overlap_pct": round(len(overlap) / max(len(golden_set), 1), 3),
        "matched": sorted(overlap),
        "missed": sorted(golden_set - actual_set),
    }


def route_check(actual_route: str, expected_route: str | None) -> dict | None:
    """Compares `response.metadata.route` against a golden's `expected_route`.

    Only set for adaptive_retrieval goldens today — retrieval being skipped on
    purpose isn't a missing source (`source_overlap` can't express it), it's the
    point, so route is the pass/fail signal for that feature instead.
    """
    if expected_route is None:
        return None
    return {
        "passed": actual_route == expected_route,
        "actual_route": actual_route,
        "expected_route": expected_route,
    }


def web_fallback_check(used_web_fallback: bool, golden_sources: list[str]) -> dict | None:
    """Compares `response.metadata.used_web_fallback` against the `tavily_web`
    sentinel goldens use in place of a real filename (CRAG goldens q-017..q-020).

    `source_overlap` can never pass for these: `golden_sources: [tavily_web]` is
    a marker, not a real ingested file, so it can never appear in `actual_sources`
    no matter how well CRAG's web fallback actually worked. This is the same
    "the normal source check can't express this outcome" situation `route_check`
    handles for adaptive_retrieval, just for a different feature.
    """
    if golden_sources != ["tavily_web"]:
        return None
    return {"passed": used_web_fallback, "used_web_fallback": used_web_fallback}
