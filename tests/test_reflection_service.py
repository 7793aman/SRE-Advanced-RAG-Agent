"""Unit seam: reflect(), should_regenerate(), and needs_retrieval(), with the
critic/classifier LLM call faked. Nothing here calls OpenAI.

Gate behaviour (app/config.py defaults):
  reflection_score >= reflection_min_score (0.85)          -> don't regenerate
  reflection_score < reflection_min_score                   -> regenerate,
    unless retries_so_far >= max_reflection_retries (2)
"""

from __future__ import annotations

import json

import pytest

from app.models import ReflectionResult, RetrievedChunk
from app.services.llm_service import LLMResponse

_CHUNKS = [
    RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=0.6),
]


def _critic_response(
    score: float, needs_regeneration: bool = False, refined_question: str = ""
) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(
            {
                "reflection_score": score,
                "needs_regeneration": needs_regeneration,
                "refined_question": refined_question,
                "reasoning": "test",
            }
        )
    )


def _gate_response(needs_retrieval: bool) -> LLMResponse:
    return LLMResponse(text=json.dumps({"needs_retrieval": needs_retrieval, "reasoning": "test"}))


# --- reflect(): the critic call ---------------------------------------------


def test_reflect_returns_the_critics_score_and_refined_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    monkeypatch.setattr(
        reflection_service,
        "generate_json",
        lambda *a, **k: _critic_response(0.4, needs_regeneration=True, refined_question="sharper"),
    )

    result = reflection_service.reflect("tell me about scaling", "some vague answer", _CHUNKS)

    assert result.reflection_score == pytest.approx(0.4)
    assert result.refined_question == "sharper"


def test_reflect_prompt_includes_the_question_answer_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    calls: list[dict] = []

    def _fake_generate_json(prompt: str, **kwargs: object) -> LLMResponse:
        calls.append({"prompt": prompt, **kwargs})
        return _critic_response(0.9)

    monkeypatch.setattr(reflection_service, "generate_json", _fake_generate_json)

    reflection_service.reflect("What is a Pod?", "A Pod is a deployable unit.", _CHUNKS)

    assert "What is a Pod?" in calls[0]["prompt"]
    assert "A Pod is a deployable unit." in calls[0]["prompt"]
    assert "pods.html" in calls[0]["prompt"]


def test_reflect_notes_when_no_context_was_retrieved(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty chunk list means Self-RAG deliberately skipped retrieval
    (general knowledge) — the critic prompt must say so plainly, rather than
    silently handing the critic an empty "Retrieved context:" section that
    reads like a failed corpus search."""
    from app.services import reflection_service

    calls: list[dict] = []

    def _fake_generate_json(prompt: str, **kwargs: object) -> LLMResponse:
        calls.append({"prompt": prompt, **kwargs})
        return _critic_response(0.9)

    monkeypatch.setattr(reflection_service, "generate_json", _fake_generate_json)

    reflection_service.reflect("What is 2 + 2?", "4", [])

    assert "general knowledge" in calls[0]["prompt"].lower()


def test_critic_call_failure_degrades_to_accepting_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    def _boom(*a: object, **k: object) -> LLMResponse:
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr(reflection_service, "generate_json", _boom)

    result = reflection_service.reflect("What is a Pod?", "answer", _CHUNKS)

    assert result.needs_regeneration is False
    assert result.reflection_score == pytest.approx(1.0)


def test_malformed_critic_json_degrades_to_accepting_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    monkeypatch.setattr(
        reflection_service, "generate_json", lambda *a, **k: LLMResponse(text="not valid json")
    )

    result = reflection_service.reflect("What is a Pod?", "answer", _CHUNKS)

    assert result.needs_regeneration is False
    assert result.reflection_score == pytest.approx(1.0)


# --- should_regenerate(): the bounded gate ----------------------------------


def test_high_score_never_regenerates() -> None:
    from app.config import settings
    from app.services.reflection_service import should_regenerate

    reflection = ReflectionResult(reflection_score=settings.reflection_min_score)

    assert should_regenerate(reflection, retries_so_far=0) is False


def test_score_exactly_at_the_threshold_does_not_regenerate() -> None:
    from app.config import settings
    from app.services.reflection_service import should_regenerate

    reflection = ReflectionResult(reflection_score=settings.reflection_min_score)

    assert should_regenerate(reflection, retries_so_far=0) is False


def test_low_score_regenerates_while_under_the_retry_ceiling() -> None:
    from app.services.reflection_service import should_regenerate

    reflection = ReflectionResult(reflection_score=0.2)

    assert should_regenerate(reflection, retries_so_far=0) is True


def test_low_score_stops_once_the_retry_ceiling_is_reached() -> None:
    from app.config import settings
    from app.services.reflection_service import should_regenerate

    reflection = ReflectionResult(reflection_score=0.2)

    assert should_regenerate(reflection, retries_so_far=settings.max_reflection_retries) is False


def test_gate_ignores_the_critics_own_needs_regeneration_opinion() -> None:
    """Matches crag_service's threshold-over-label pattern: the score against
    the fixed threshold decides, not the LLM's own boolean verdict."""
    from app.services.reflection_service import should_regenerate

    reflection = ReflectionResult(reflection_score=0.99, needs_regeneration=True)

    assert should_regenerate(reflection, retries_so_far=0) is False


# --- needs_retrieval(): the adaptive-retrieval gate -------------------------


def test_needs_retrieval_true_when_the_classifier_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    monkeypatch.setattr(reflection_service, "generate_json", lambda *a, **k: _gate_response(True))

    assert reflection_service.needs_retrieval("Why is my pod OOMKilled?") is True


def test_needs_retrieval_false_for_general_knowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import reflection_service

    monkeypatch.setattr(reflection_service, "generate_json", lambda *a, **k: _gate_response(False))

    assert reflection_service.needs_retrieval("What is 2 + 2?") is False


def test_classifier_call_failure_degrades_to_needing_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    def _boom(*a: object, **k: object) -> LLMResponse:
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr(reflection_service, "generate_json", _boom)

    assert reflection_service.needs_retrieval("What is 2 + 2?") is True


def test_malformed_classifier_json_degrades_to_needing_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import reflection_service

    monkeypatch.setattr(
        reflection_service, "generate_json", lambda *a, **k: LLMResponse(text="not valid json")
    )

    assert reflection_service.needs_retrieval("What is 2 + 2?") is True
