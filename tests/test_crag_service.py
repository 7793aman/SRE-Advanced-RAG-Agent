"""Unit seam: evaluate_and_correct(), with the grader LLM call and Tavily web
search both faked. Nothing here calls OpenAI or Tavily.

Threshold behaviour (app/config.py defaults):
  score >= crag_relevance_threshold (0.7)                  -> correct, no web call
  crag_ambiguous_threshold (0.5) <= score < 0.7             -> ambiguous, corpus + web
  score < crag_ambiguous_threshold (0.5)                    -> incorrect, web only
"""

from __future__ import annotations

import json

import pytest

from app.models import RetrievedChunk
from app.services.llm_service import LLMResponse
from app.services.web_search_service import WebSearchUnconfiguredError

_CHUNKS = [
    RetrievedChunk(text="A Pod is the smallest deployable unit.", source="pods.html", score=0.6),
]
_WEB_CHUNKS = [
    RetrievedChunk(
        text="Kubernetes v1.32 is the latest stable release.",
        source="https://kubernetes.io/releases/",
        score=0.9,
    )
]


def _grade_response(score: float, label: str = "") -> LLMResponse:
    return LLMResponse(
        text=json.dumps(
            {
                "relevance_score": score,
                "relevance_label": label,
                "confidence": 0.8,
                "reasoning": "test",
            }
        )
    )


# --- thresholds: correct / ambiguous / incorrect ----------------------------


def test_high_score_keeps_the_chunks_unchanged_and_never_calls_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.9))
    monkeypatch.setattr(
        crag_service,
        "web_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("web_search should not be called")),
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


def test_the_grading_itself_is_attached_to_the_result_not_just_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_and_correct used to compute a real relevance_score/label and
    only log it (`logger.info(...)`), throwing it away before it could ever
    reach a caller — so nothing outside the server logs could see why a
    correction did or didn't fire. Every branch (correct/ambiguous/incorrect)
    must attach the same CRAGEvaluation the threshold decision was made from."""
    from app.services import crag_service

    monkeypatch.setattr(
        crag_service, "generate_json", lambda *a, **k: _grade_response(0.9, "correct")
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.evaluation is not None
    assert result.evaluation.relevance_score == pytest.approx(0.9)
    assert result.evaluation.relevance_label == "correct"


def test_low_score_below_ambiguous_threshold_discards_the_corpus_for_web_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.2))
    monkeypatch.setattr(crag_service, "web_search", lambda *a, **k: list(_WEB_CHUNKS))

    result = crag_service.evaluate_and_correct(
        "What's the latest stable Kubernetes release?", _CHUNKS
    )

    assert result.chunks == _WEB_CHUNKS
    assert result.used_web_fallback is True


def test_mid_score_in_the_ambiguous_band_merges_corpus_and_web_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.6))
    monkeypatch.setattr(crag_service, "web_search", lambda *a, **k: list(_WEB_CHUNKS))

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS + _WEB_CHUNKS
    assert result.used_web_fallback is True


def test_score_exactly_at_the_relevance_threshold_counts_as_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings
    from app.services import crag_service

    monkeypatch.setattr(
        crag_service,
        "generate_json",
        lambda *a, **k: _grade_response(settings.crag_relevance_threshold),
    )
    monkeypatch.setattr(
        crag_service,
        "web_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("web_search should not be called")),
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


# --- top_k: the caller's chunk count, not Tavily's own default -------------


def test_incorrect_grade_requests_and_caps_web_results_at_the_callers_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    ten_web_chunks = [
        RetrievedChunk(
            text=f"web result {i}", source=f"https://example.com/{i}", score=0.9 - i / 100
        )
        for i in range(10)
    ]
    calls: list[dict] = []
    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.1))
    monkeypatch.setattr(
        crag_service,
        "web_search",
        lambda question, max_results=5: (
            calls.append({"max_results": max_results}) or ten_web_chunks
        ),
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS, top_k=3)

    assert calls[0]["max_results"] == 3
    assert result.chunks == ten_web_chunks[:3]
    assert result.used_web_fallback is True


def test_ambiguous_grade_caps_the_merged_result_at_the_callers_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.6))
    monkeypatch.setattr(crag_service, "web_search", lambda *a, **k: list(_WEB_CHUNKS))

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS, top_k=1)

    assert result.chunks == (_CHUNKS + _WEB_CHUNKS)[:1]
    assert result.used_web_fallback is True


# --- empty retrieval: skip the grader entirely ------------------------------


def test_empty_retrieval_skips_the_grader_and_goes_straight_to_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(
        crag_service,
        "generate_json",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("grader should not be called")),
    )
    monkeypatch.setattr(crag_service, "web_search", lambda *a, **k: list(_WEB_CHUNKS))

    result = crag_service.evaluate_and_correct("What is a Pod?", [])

    assert result.chunks == _WEB_CHUNKS
    assert result.used_web_fallback is True


# --- graceful degradation: a broken/unconfigured web fallback never crashes -


def test_unconfigured_web_search_degrades_to_the_original_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.1))

    def _boom(*a: object, **k: object) -> list[RetrievedChunk]:
        raise WebSearchUnconfiguredError("TAVILY_API_KEY is not configured")

    monkeypatch.setattr(crag_service, "web_search", _boom)

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


def test_grader_call_failure_skips_correction_and_never_calls_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken judge isn't evidence the retrieval is bad — a grader outage
    must degrade to leaving the chunks alone, not to assuming they're wrong
    and spending a Tavily call on every request while OpenAI is down."""
    from app.services import crag_service

    monkeypatch.setattr(
        crag_service,
        "generate_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OpenAI is down")),
    )
    monkeypatch.setattr(
        crag_service,
        "web_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("web_search should not be called")),
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


def test_web_search_failure_also_degrades_to_the_original_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.1))
    monkeypatch.setattr(
        crag_service,
        "web_search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Tavily API timed out")),
    )

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


def test_unconfigured_web_search_on_empty_retrieval_degrades_to_an_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty retrieval + no web fallback available: nothing crashes, and
    there's honestly nothing to answer from — an empty list, same as any
    other zero-chunk retrieval the rest of the pipeline already handles."""
    from app.services import crag_service

    def _boom(*a: object, **k: object) -> list[RetrievedChunk]:
        raise WebSearchUnconfiguredError("TAVILY_API_KEY is not configured")

    monkeypatch.setattr(crag_service, "web_search", _boom)

    result = crag_service.evaluate_and_correct("What is a Pod?", [])

    assert result.chunks == []
    assert result.used_web_fallback is False


# --- malformed grader output: degrade like a call failure, don't crash -----


def test_malformed_grader_json_keeps_retrieval_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same degradation as a grader call failure: a broken judge isn't
    evidence the retrieval is bad, so this must NOT force a web fallback on
    perfectly good chunks (regression: it used to return "incorrect" here,
    contradicting the module's own documented contract)."""
    from app.services import crag_service

    def _boom(*args: object, **kwargs: object) -> list:
        raise AssertionError("malformed grader JSON must not trigger a web search")

    monkeypatch.setattr(
        crag_service, "generate_json", lambda *a, **k: LLMResponse(text="not valid json")
    )
    monkeypatch.setattr(crag_service, "web_search", _boom)

    result = crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert result.chunks == _CHUNKS
    assert result.used_web_fallback is False


# --- the grading prompt -----------------------------------------------------


def test_grading_prompt_includes_the_question_and_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import crag_service

    calls: list[dict] = []

    def _fake_generate_json(prompt: str, **kwargs: object) -> LLMResponse:
        calls.append({"prompt": prompt, **kwargs})
        return _grade_response(0.9)

    monkeypatch.setattr(crag_service, "generate_json", _fake_generate_json)

    crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)

    assert "What is a Pod?" in calls[0]["prompt"]
    assert "pods.html" in calls[0]["prompt"]
    assert "A Pod is the smallest deployable unit." in calls[0]["prompt"]


# --- logging: the grade is always logged, per the issue's scope ------------


def test_grade_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from loguru import logger

    from app.services import crag_service

    monkeypatch.setattr(crag_service, "generate_json", lambda *a, **k: _grade_response(0.9))

    messages: list[str] = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO")
    try:
        crag_service.evaluate_and_correct("What is a Pod?", _CHUNKS)
    finally:
        logger.remove(handler_id)

    assert any("0.9" in m or "0.90" in m for m in messages)
