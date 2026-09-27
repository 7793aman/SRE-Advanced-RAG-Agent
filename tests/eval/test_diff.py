"""Unit seam: comparing two already-written result JSON files. `make eval-diff`
(issue #33, story 57) needs `all` to score at least as high as `naive` on
faithfulness + context recall, and every `expected_baseline: fail` golden to
actually flip to pass under `all` — this is the module the reference tutorial
never had (it only ever printed an aggregate table per run, never diffed two).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.diff import compare_runs, find_latest, main

_NAIVE = {
    "profile": "naive",
    "aggregate": {"faithfulness": 0.6, "context_recall": 0.55, "passed": 2, "failed": 2},
    "rows": [
        {"id": "q-001", "expected_baseline": "pass"},
        {"id": "q-005", "expected_baseline": "fail"},
    ],
}

_ALL_BETTER = {
    "profile": "all",
    "aggregate": {"faithfulness": 0.8, "context_recall": 0.75, "passed": 4, "failed": 0},
    "expected_outcome_mismatches": [],
    "rows": [
        {"id": "q-001", "expected_with_feature": "pass"},
        {"id": "q-005", "expected_with_feature": "pass"},
    ],
}


def test_compare_runs_ok_when_all_scores_higher_and_no_mismatches() -> None:
    result = compare_runs(_NAIVE, _ALL_BETTER)
    assert result["ok"] is True
    assert result["regressions"] == []
    assert result["faithfulness_delta"] == pytest.approx(0.2)
    assert result["context_recall_delta"] == pytest.approx(0.2)


def test_compare_runs_treats_a_none_metric_as_uncomparable_not_a_crash() -> None:
    # aggregate() writes None when a run has zero scoreable rows (e.g. every
    # golden errored or was skipped) — must not raise, and must not count as ok.
    empty_all = {**_ALL_BETTER, "aggregate": {**_ALL_BETTER["aggregate"], "faithfulness": None}}
    result = compare_runs(_NAIVE, empty_all)
    assert result["ok"] is False
    assert result["faithfulness_delta"] is None
    assert any("cannot be compared" in r for r in result["regressions"])


def test_main_does_not_crash_when_a_metric_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "20260101T000000Z_naive.json").write_text(json.dumps(_NAIVE))
    empty_all = {**_ALL_BETTER, "aggregate": {**_ALL_BETTER["aggregate"], "faithfulness": None}}
    (tmp_path / "20260101T000000Z_all.json").write_text(json.dumps(empty_all))

    monkeypatch.setattr("sys.argv", ["diff", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_compare_runs_flags_a_faithfulness_regression() -> None:
    worse = {**_ALL_BETTER, "aggregate": {**_ALL_BETTER["aggregate"], "faithfulness": 0.5}}
    result = compare_runs(_NAIVE, worse)
    assert result["ok"] is False
    assert any("faithfulness" in r for r in result["regressions"])


def test_compare_runs_flags_a_context_recall_regression() -> None:
    worse = {**_ALL_BETTER, "aggregate": {**_ALL_BETTER["aggregate"], "context_recall": 0.1}}
    result = compare_runs(_NAIVE, worse)
    assert result["ok"] is False
    assert any("context_recall" in r for r in result["regressions"])


def test_compare_runs_flags_an_expected_baseline_fail_golden_that_did_not_flip() -> None:
    still_failing = {
        **_ALL_BETTER,
        "expected_outcome_mismatches": [{"id": "q-005", "expected": "pass", "actual": "fail"}],
    }
    result = compare_runs(_NAIVE, still_failing)
    assert result["ok"] is False
    assert any("q-005" in r for r in result["regressions"])


def test_find_latest_picks_the_newest_matching_timestamped_file(tmp_path: Path) -> None:
    (tmp_path / "20260101T000000Z_naive.json").write_text("{}")
    newest = tmp_path / "20260901T000000Z_naive.json"
    newest.write_text("{}")
    (tmp_path / "20260101T000000Z_all.json").write_text("{}")

    assert find_latest(tmp_path, "naive") == newest


def test_find_latest_raises_a_clear_error_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="naive"):
        find_latest(tmp_path, "naive")


def test_main_exits_nonzero_on_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    (tmp_path / "20260101T000000Z_naive.json").write_text(json.dumps(_NAIVE))
    worse = {**_ALL_BETTER, "aggregate": {**_ALL_BETTER["aggregate"], "faithfulness": 0.1}}
    (tmp_path / "20260101T000000Z_all.json").write_text(json.dumps(worse))

    monkeypatch.setattr("sys.argv", ["diff", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "faithfulness" in capsys.readouterr().out


def test_main_exits_zero_when_all_is_at_least_as_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "20260101T000000Z_naive.json").write_text(json.dumps(_NAIVE))
    (tmp_path / "20260101T000000Z_all.json").write_text(json.dumps(_ALL_BETTER))

    monkeypatch.setattr("sys.argv", ["diff", "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
