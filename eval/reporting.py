"""Turns per-golden checks into a single PASS/FAIL, and diffs that against what
the golden's YAML said to expect (issue #33, stories 56-57).

The reference tutorial's `expected_baseline`/`expected_with_feature` fields were
validated on load but never read by anything downstream — this module is what
was missing. `golden_passed()` is deliberately built only from the deterministic
checks (forbidden keywords, source overlap, route), never from a raw Ragas score:
Ragas is an LLM judge and its numbers are noisy — see
`learning/lessons/0010-one-golden-end-to-end.html`'s caution box, and
`arxiv.org/pdf/2510.27106`. Ragas metrics are still recorded and averaged per run;
they're just not what a golden's PASS/FAIL hinges on.
"""

from __future__ import annotations

from statistics import mean

METRIC_KEYS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
]


def golden_passed(row: dict) -> bool:
    if not row["forbidden_check"]["passed"]:
        return False
    route_check = row.get("route_check")
    if route_check is not None:
        return bool(route_check["passed"])
    web_fallback_check = row.get("web_fallback_check")
    if web_fallback_check is not None:
        return bool(web_fallback_check["passed"])
    return row["source_overlap"]["overlap_pct"] >= 1.0


def expected_outcome_mismatches(rows: list[dict], profile: str) -> list[dict]:
    """Compares each row's actual PASS/FAIL to what its golden expected.

    `naive` is checked against `expected_baseline`; every other profile is checked
    against `expected_with_feature` (always "pass"). This is exact for `naive` vs
    `all` (issue #33's actual comparison — `all` turns on every technique, so it
    is "the golden's own technique's profile" for every golden simultaneously);
    for a single-technique profile like `hybrid+rerank`, this also flags a
    baseline-tagged golden that regressed, which is a reasonable bonus signal
    even though the profile isn't literally that golden's own technique.
    """
    expected_field = "expected_baseline" if profile == "naive" else "expected_with_feature"
    mismatches = []
    for row in rows:
        expected = row[expected_field]
        actual = "pass" if golden_passed(row) else "fail"
        if actual != expected:
            mismatches.append({"id": row["id"], "expected": expected, "actual": actual})
    return mismatches


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for k in METRIC_KEYS:
        vals = [
            r["ragas_metrics"].get(k)
            for r in rows
            if r.get("ragas_metrics") and r["ragas_metrics"].get(k) is not None
        ]
        out[k] = round(mean(vals), 3) if vals else None
    out["forbidden_violations"] = sum(
        1 for r in rows if not r.get("forbidden_check", {}).get("passed", True)
    )
    out["passed"] = sum(1 for r in rows if golden_passed(r))
    out["failed"] = len(rows) - out["passed"]
    return out


def print_table(payload: dict) -> None:
    """Prints a markdown table of per-question results + the aggregate row."""
    print(f"\n## Eval — profile={payload['profile']} mode={payload['mode']}")
    print(f"Skipped: {len(payload['skipped'])}")
    print()
    print("| id | feature | faith | ctx_prec | ctx_recall | ans_rel | forbidden | pass? |")
    print("|----|---------|-------|----------|------------|---------|-----------|-------|")

    for r in payload["rows"]:
        m = r.get("ragas_metrics") or {}
        fb = "OK" if r["forbidden_check"]["passed"] else f"FAIL: {r['forbidden_check']['hits']}"
        verdict = "PASS" if golden_passed(r) else "FAIL"
        print(
            f"| {r['id']} | {r['demonstrates_feature']} | "
            f"{m.get('faithfulness', 0):.2f} | "
            f"{m.get('context_precision', 0):.2f} | "
            f"{m.get('context_recall', 0):.2f} | "
            f"{m.get('answer_relevancy', 0):.2f} | {fb} | {verdict} |"
        )

    if not payload["rows"]:
        print("(no rows evaluated — all goldens skipped)")
        return

    a = payload["aggregate"]
    print(
        f"| **AGG** | — | **{a['faithfulness']}** | "
        f"**{a['context_precision']}** | **{a['context_recall']}** | "
        f"**{a['answer_relevancy']}** | "
        f"violations={a['forbidden_violations']} | {a['passed']}/{a['passed'] + a['failed']} |"
    )

    mismatches = payload.get("expected_outcome_mismatches", [])
    if mismatches:
        print(f"\n**{len(mismatches)} golden(s) did not match their expected outcome:**")
        for m in mismatches:
            print(f"  - {m['id']}: expected {m['expected']}, got {m['actual']}")
