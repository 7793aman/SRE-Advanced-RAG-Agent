"""`make eval-diff` (issue #33, story 57): compares the latest `naive` and `all`
result files. Passes only if `all` scores at least as high as `naive` on
faithfulness and context recall, AND every golden tagged `expected_baseline: fail`
actually flips to pass under `all` (i.e. `all`'s own `expected_outcome_mismatches`
is empty). This module didn't exist in the reference tutorial — it only ever
printed one run's aggregate table, never diffed two — but the Makefile
(`eval-diff: uv run python -m eval.diff`) and issue #33's "Done when" checklist
both need it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_GATED_METRICS = ("faithfulness", "context_recall")


def find_latest(results_dir: Path, profile: str) -> Path:
    matches = sorted(results_dir.glob(f"*_{profile}.json"))
    if not matches:
        raise FileNotFoundError(f"No result file found for profile={profile!r} in {results_dir}")
    return matches[-1]


def compare_runs(naive: dict, all_run: dict) -> dict:
    regressions: list[str] = []
    deltas: dict[str, float | None] = {}

    for metric in _GATED_METRICS:
        naive_val = naive["aggregate"][metric]
        all_val = all_run["aggregate"][metric]
        # aggregate() legitimately writes None when a run has zero scoreable rows
        # (e.g. every golden errored or was skipped) — that can't be diffed, and
        # it isn't a pass either.
        if naive_val is None or all_val is None:
            deltas[f"{metric}_delta"] = None
            regressions.append(
                f"{metric} cannot be compared: naive={naive_val} all={all_val} "
                "(one run has no scoreable rows)"
            )
            continue
        deltas[f"{metric}_delta"] = all_val - naive_val
        if all_val < naive_val:
            regressions.append(f"{metric} regressed: naive={naive_val:.3f} all={all_val:.3f}")

    for mismatch in all_run.get("expected_outcome_mismatches", []):
        regressions.append(
            f"{mismatch['id']} still {mismatch['actual']} under 'all' "
            f"(expected {mismatch['expected']})"
        )

    return {"ok": not regressions, "regressions": regressions, **deltas}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff the naive vs all eval runs")
    ap.add_argument("--results-dir", default="eval/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    naive = json.loads(find_latest(results_dir, "naive").read_text())
    all_run = json.loads(find_latest(results_dir, "all").read_text())

    result = compare_runs(naive, all_run)

    def _fmt_delta(value: float | None) -> str:
        return f"{value:+.3f}" if value is not None else "n/a"

    print("## Eval diff — naive vs all")
    print(
        f"faithfulness:     naive={naive['aggregate']['faithfulness']}"
        f"  all={all_run['aggregate']['faithfulness']}"
        f"  delta={_fmt_delta(result['faithfulness_delta'])}"
    )
    print(
        f"context_recall:   naive={naive['aggregate']['context_recall']}"
        f"  all={all_run['aggregate']['context_recall']}"
        f"  delta={_fmt_delta(result['context_recall_delta'])}"
    )

    if result["ok"]:
        print(
            "\nPASS — 'all' matches or beats 'naive', every expected-fail golden flipped to pass."
        )
        sys.exit(0)
    else:
        print(f"\nFAIL — {len(result['regressions'])} regression(s):")
        for r in result["regressions"]:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
