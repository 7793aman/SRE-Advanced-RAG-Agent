"""Unit seam: the golden-set loader and its validation rules. Pure — no I/O
beyond reading a YAML file we write to a tmp_path, no network, no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eval.schema import Golden, load_goldens

_VALID = {
    "id": "q-001",
    "question": "What is a Pod?",
    "intent": "rag",
    "golden_sources": ["concepts__workloads__pods.html"],
    "golden_answer_keywords": ["pod", "container"],
    "demonstrates_feature": "baseline",
    "expected_baseline": "pass",
    "expected_with_feature": "pass",
    "notes": "control case",
}


def test_valid_golden_parses() -> None:
    g = Golden.model_validate(_VALID)
    assert g.id == "q-001"
    assert g.forbidden_keywords == []


def test_id_must_match_q_nnn_pattern() -> None:
    with pytest.raises(ValidationError):
        Golden.model_validate({**_VALID, "id": "not-an-id"})


def test_expected_with_feature_must_be_pass() -> None:
    with pytest.raises(ValidationError):
        Golden.model_validate({**_VALID, "expected_with_feature": "fail"})


def test_golden_sources_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        Golden.model_validate({**_VALID, "golden_sources": []})


def test_adaptive_retrieval_is_a_valid_feature() -> None:
    g = Golden.model_validate({**_VALID, "demonstrates_feature": "adaptive_retrieval"})
    assert g.demonstrates_feature == "adaptive_retrieval"


def test_forbidden_keywords_defaults_to_empty_list() -> None:
    assert Golden.model_validate(_VALID).forbidden_keywords == []


def test_forbidden_keywords_can_be_set() -> None:
    g = Golden.model_validate({**_VALID, "forbidden_keywords": ["kubeconfig"]})
    assert g.forbidden_keywords == ["kubeconfig"]


def test_load_goldens_reads_a_list_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "goldens.yaml"
    path.write_text(yaml.safe_dump([_VALID, {**_VALID, "id": "q-002"}]))
    goldens = load_goldens(path)
    assert [g.id for g in goldens] == ["q-001", "q-002"]


def test_load_goldens_rejects_a_non_list_root(tmp_path: Path) -> None:
    path = tmp_path / "goldens.yaml"
    path.write_text(yaml.safe_dump({"not": "a list"}))
    with pytest.raises(ValueError, match="Expected YAML root to be a list"):
        load_goldens(path)


def test_load_goldens_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "goldens.yaml"
    path.write_text(yaml.safe_dump([_VALID, _VALID]))
    with pytest.raises(ValueError, match="Duplicate golden IDs"):
        load_goldens(path)


def test_load_goldens_warns_but_does_not_fail_on_missing_feature_categories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "goldens.yaml"
    path.write_text(yaml.safe_dump([_VALID]))
    with pytest.warns(UserWarning, match="No golden entries for features"):
        goldens = load_goldens(path)
    assert len(goldens) == 1


def test_the_projects_real_golden_set_loads_cleanly() -> None:
    real_path = Path(__file__).resolve().parents[2] / "eval" / "seed_questions.yaml"
    goldens = load_goldens(real_path)
    assert len(goldens) == 42
    ids = [g.id for g in goldens]
    assert ids == sorted(ids)
