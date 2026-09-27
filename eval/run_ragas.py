"""CLI entry point (issue #33, stories 56 & 58): `make eval-baseline` / `make
eval-all` both run `python -m eval.run_ragas --profile <name>`.

For each golden: ask the real pipeline the question (`ServiceInvoker`), run the
cheap deterministic checks immediately, score the retrieval-based rows with
Ragas, and write one timestamped JSON file with everything plus a PASS/FAIL
verdict per golden (`eval.reporting.golden_passed`) compared against what the
YAML expected.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from app.models import ChatResponse, RetrievedChunk
from eval.invokers import ServiceInvoker, SkippedIntent
from eval.post_checks import (
    forbidden_keywords_check,
    route_check,
    source_overlap,
    web_fallback_check,
)
from eval.profiles import PROFILES
from eval.ragas_adapter import run as run_ragas_metrics
from eval.reporting import aggregate, expected_outcome_mismatches, print_table
from eval.schema import Golden, load_goldens


def build_row(golden: Golden, resp: ChatResponse, chunks: list[RetrievedChunk]) -> dict:
    contexts = [c.text for c in chunks]
    actual_sources = resp.sources
    return {
        "id": golden.id,
        "demonstrates_feature": golden.demonstrates_feature,
        "intent": golden.intent,
        "question": golden.question,
        "answer": resp.answer,
        "contexts": contexts,
        "ground_truth": ", ".join(golden.golden_answer_keywords),
        "actual_sources": actual_sources,
        "golden_sources": golden.golden_sources,
        "expected_baseline": golden.expected_baseline,
        "expected_with_feature": golden.expected_with_feature,
        "forbidden_check": forbidden_keywords_check(resp.answer, golden.forbidden_keywords),
        "source_overlap": source_overlap(actual_sources, golden.golden_sources),
        "route_check": route_check(resp.metadata.route, golden.expected_route),
        "web_fallback_check": web_fallback_check(
            resp.metadata.used_web_fallback, golden.golden_sources
        ),
    }


def run_eval(goldens: list[Golden], flags: dict, invoker: Any) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skipped: list[dict] = []
    for g in goldens:
        try:
            resp, chunks = invoker.invoke(g.question, flags, g.intent)
        except SkippedIntent as e:
            skipped.append({"id": g.id, "reason": str(e)})
            continue
        except Exception as e:  # noqa: BLE001 — one bad golden must not kill the run
            skipped.append({"id": g.id, "reason": f"error: {e}"})
            continue
        rows.append(build_row(g, resp, chunks))
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Ragas eval harness")
    ap.add_argument(
        "--profile", required=True, choices=list(PROFILES.keys()), help="Flag profile to evaluate"
    )
    ap.add_argument(
        "--questions", default="eval/seed_questions.yaml", help="Path to the golden set YAML"
    )
    ap.add_argument(
        "--filter",
        default=None,
        help="Only run goldens with demonstrates_feature == FILTER (plus baseline)",
    )
    ap.add_argument("--mode", default="service", choices=["service", "api"], help="Invocation mode")
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/<timestamp>_<profile>.json)",
    )
    args = ap.parse_args()

    flags = PROFILES[args.profile]
    goldens = load_goldens(args.questions)
    if args.filter:
        goldens = [g for g in goldens if g.demonstrates_feature in (args.filter, "baseline")]

    if args.mode == "service":
        invoker = ServiceInvoker()
    else:
        print(
            "API mode not yet implemented — sql/hybrid_rag_sql goldens need the "
            "graph's SQL-approval interrupt over real HTTP, not just a service call.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows, skipped = run_eval(goldens, flags, invoker)

    # "error: ..." skips are unexpected exceptions (a real bug), not the expected,
    # structural skips (unsupported intent, tavily unset). aggregate()/golden_passed()
    # only ever look at `rows`, so a run where most goldens errored out could
    # otherwise still print a clean 100%-pass table — surface this loudly instead
    # of leaving it buried in the JSON's `skipped` list.
    error_skips = [s for s in skipped if s["reason"].startswith("error:")]
    if error_skips:
        print(
            f"WARNING: {len(error_skips)} golden(s) errored (not just skipped) and are "
            "excluded from this run's pass/fail numbers:",
            file=sys.stderr,
        )
        for s in error_skips:
            print(f"  - {s['id']}: {s['reason']}", file=sys.stderr)

    # Ragas needs real retrieved context to judge against. A row where retrieval
    # was correctly skipped (adaptive_retrieval's `expected_route` case) has none
    # — score only rows that actually retrieved something.
    scoreable = [r for r in rows if r["contexts"]]
    metrics = run_ragas_metrics(scoreable) if scoreable else []
    if len(metrics) != len(scoreable):
        # A judge-LLM error/rate-limit on one sample can make Ragas return fewer
        # records than it was given. Don't let that crash the run and discard
        # every already-computed row — warn, and score only however many lined up.
        print(
            f"WARNING: ragas returned {len(metrics)} records for {len(scoreable)} "
            "scoreable rows; some rows will be left unscored.",
            file=sys.stderr,
        )
    for row, m in zip(scoreable, metrics):  # noqa: B905 — mismatch handled above, not fatal
        row["ragas_metrics"] = m
    for row in rows:
        row.setdefault("ragas_metrics", None)

    timestamp = datetime.datetime.now(datetime.UTC)
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"eval/results/{timestamp:%Y%m%dT%H%M%SZ}_{args.profile}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile": args.profile,
        "flags": flags,
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "filter": args.filter,
        "mode": args.mode,
        "rows": rows,
        "skipped": skipped,
        "aggregate": aggregate(rows),
        "expected_outcome_mismatches": expected_outcome_mismatches(rows, args.profile),
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print_table(payload)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
