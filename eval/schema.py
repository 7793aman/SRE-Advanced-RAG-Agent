"""Golden-set schema and loader (issue #33, story 55).

A golden is one test case: a question, the technique it's designed to demonstrate,
which docs a correct retrieval must pull in, and whether the naive baseline is
expected to pass or fail it. See eval/seed_questions.yaml and the project's
`learning/lessons/0010-one-golden-end-to-end.html` for a worked walkthrough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

INTENT = Literal["rag", "sql", "hybrid", "web_fallback"]
FEATURE = Literal[
    "baseline",
    "sparse",
    "dense",
    "hybrid",
    "rerank",
    "hyde",
    "crag",
    "self_rag",
    "sql",
    "hybrid_rag_sql",
    "security",
    "wild",
    "adaptive_retrieval",
]


class Golden(BaseModel):
    id: str = Field(..., pattern=r"^q-\d{3}$")
    question: str = Field(..., min_length=1)
    intent: INTENT
    golden_sources: list[str] = Field(..., min_length=1)
    golden_answer_keywords: list[str] = Field(..., min_length=1)
    demonstrates_feature: FEATURE
    expected_baseline: Literal["pass", "fail"]
    # Every golden must be solvable by *some* profile — this project's Makefile
    # gate (`make eval-diff`) only checks that expected-fail goldens pass once
    # their technique is on, so a golden that can never pass would silently
    # never be checked. Enforced by the Literal itself (Pydantic rejects any
    # other value at parse time) — no separate validator needed for one value.
    expected_with_feature: Literal["pass"]
    notes: str
    forbidden_keywords: list[str] = Field(default_factory=list)
    expected_route: str | None = None
    """Checked against `response.metadata.route` when set (`eval.post_checks.route_check`).
    Only `adaptive_retrieval` goldens use this today — it's the one thing that
    distinguishes "correctly skipped retrieval" from "correctly retrieved," which
    `golden_sources`/`source_overlap` can't express (retrieval being skipped on
    purpose isn't a missing source, it's the point)."""


def load_goldens(path: str | Path) -> list[Golden]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected YAML root to be a list, got {type(raw).__name__}")

    goldens = [Golden.model_validate(entry) for entry in raw]

    ids = [g.id for g in goldens]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"Duplicate golden IDs found: {duplicates}")

    # Warn (don't fail) if some feature categories have no entries. Not every
    # feature needs eval goldens — e.g. dense/sparse/security are demo-only and
    # intentionally excluded from the eval-progression set.
    present_features = {g.demonstrates_feature for g in goldens}
    all_features = set(FEATURE.__args__)  # type: ignore[attr-defined]
    missing = all_features - present_features
    if missing:
        import warnings

        warnings.warn(f"No golden entries for features: {missing} (OK if demo-only)", stacklevel=2)

    return goldens
