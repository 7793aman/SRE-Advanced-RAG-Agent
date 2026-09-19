"""Unit seam: the compiled graph's routing and node behaviour, against an
in-memory checkpointer. `router_service.classify_intent` and
`rag_service.run_rag` are faked — nothing here calls OpenAI, Qdrant, or
Postgres.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.models import ChatResponse, ResponseMetadata
from app.services import graph as graph_module


@pytest.fixture
def compiled_graph():  # noqa: ANN201
    return graph_module.build_graph(MemorySaver())


def _invoke(compiled_graph, question: str, flags: dict | None = None, thread_id: str = "t1"):
    return compiled_graph.invoke(
        {"question": question, "flags": flags or {}},
        {"configurable": {"thread_id": thread_id}},
    )


def _rag_response(answer: str = "A Pod is the smallest deployable unit. [pods.html]") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["pods.html"],
        retrieval_score=0.9,
        metadata=ResponseMetadata(route="rag"),
    )


# --- routing -------------------------------------------------------------------


def test_rag_intent_routes_through_retrieve_rag_to_a_grounded_answer(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "rag")
    monkeypatch.setattr(graph_module, "run_rag", lambda question, flags: _rag_response())

    result = _invoke(compiled_graph, "What is a Pod?")

    assert result["intent"] == "rag"
    assert result["response"].answer == "A Pod is the smallest deployable unit. [pods.html]"
    assert result["response"].metadata.route == "rag"


def test_sql_intent_does_not_run_the_rag_pipeline(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    def _boom(question: str, flags: dict) -> ChatResponse:
        raise AssertionError("a sql-intent question must not run the rag pipeline")

    monkeypatch.setattr(graph_module, "run_rag", _boom)

    result = _invoke(compiled_graph, "how many P1 incidents last month?")

    assert result["intent"] == "sql"
    assert result["response"].metadata.route == "sql_not_supported"


def test_hybrid_intent_does_not_run_the_rag_pipeline(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "hybrid")

    def _boom(question: str, flags: dict) -> ChatResponse:
        raise AssertionError("a hybrid-intent question must not run the rag pipeline")

    monkeypatch.setattr(graph_module, "run_rag", _boom)

    result = _invoke(compiled_graph, "list P1 incidents with remediation docs")

    assert result["intent"] == "hybrid"
    assert result["response"].metadata.route == "hybrid_not_supported"


def test_unsupported_intent_response_has_no_sources_or_pending_sql(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")
    monkeypatch.setattr(graph_module, "run_rag", lambda question, flags: _rag_response())

    result = _invoke(compiled_graph, "how many P1 incidents last month?")

    assert result["response"].sources == []
    assert result["response"].pending_sql is None


# --- state & wiring --------------------------------------------------------------


def test_question_and_flags_survive_untouched_through_every_node(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "rag")
    monkeypatch.setattr(graph_module, "run_rag", lambda question, flags: _rag_response())

    result = _invoke(compiled_graph, "What is a Pod?", flags={"search_mode": "hybrid"})

    assert result["question"] == "What is a Pod?"
    assert result["flags"] == {"search_mode": "hybrid"}


def test_run_rag_receives_the_question_and_flags(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "rag")
    calls: list[dict] = []

    def _fake_run_rag(question: str, flags: dict) -> ChatResponse:
        calls.append({"question": question, "flags": flags})
        return _rag_response()

    monkeypatch.setattr(graph_module, "run_rag", _fake_run_rag)

    _invoke(compiled_graph, "What is a Pod?", flags={"top_k": 3})

    assert calls == [{"question": "What is a Pod?", "flags": {"top_k": 3}}]


def test_two_different_threads_do_not_share_state(
    compiled_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "rag")
    monkeypatch.setattr(
        graph_module, "run_rag", lambda question, flags: _rag_response(answer=question)
    )

    first = _invoke(compiled_graph, "question one", thread_id="thread-a")
    second = _invoke(compiled_graph, "question two", thread_id="thread-b")

    assert first["response"].answer == "question one"
    assert second["response"].answer == "question two"
