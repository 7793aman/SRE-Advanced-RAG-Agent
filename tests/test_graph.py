"""Unit seam: the compiled graph's routing and node behaviour, against an
in-memory checkpointer. `router_service.classify_intent`,
`rag_service.run_rag`, and the SQL service are faked — nothing here calls
OpenAI, Qdrant, or Postgres.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.models import ChatResponse, ResponseMetadata
from app.services import graph as graph_module
from app.services.llm_service import LLMResponse
from app.services.sql_service import (
    GeneratedSQL,
    SQLExecutionError,
    SQLGenerationError,
    UnsafeSQLError,
)


@pytest.fixture
def compiled_graph():  # noqa: ANN201
    return graph_module.build_graph(MemorySaver())


def _invoke(compiled_graph, question: str, flags: dict | None = None, thread_id: str = "t1"):
    return compiled_graph.invoke(
        {"question": question, "flags": flags or {}},
        {"configurable": {"thread_id": thread_id}},
    )


def _rag_response(
    answer: str = "A Pod is the smallest deployable unit. [pods.html]",
) -> ChatResponse:
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


# --- Text2SQL + human-in-the-loop approval (issue #30) -------------------------------


@pytest.fixture
def sql_env(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake the SQL service and answer LLM; records every call."""
    env: dict = {"executed": [], "answer_prompts": [], "rows": [{"cluster": "prod-eu", "n": 7}]}

    monkeypatch.setattr(
        graph_module,
        "generate_sql",
        lambda question: GeneratedSQL(
            sql="SELECT cluster, count(*) AS n FROM incidents GROUP BY cluster",
            explanation="Counts incidents per cluster.",
        ),
    )

    def _execute(sql: str) -> tuple[list[dict], bool]:
        env["executed"].append(sql)
        return env["rows"], env.get("cache_hit", False)

    def _generate_text(prompt: str, system_prompt: str | None = None, **_: object) -> LLMResponse:
        env["answer_prompts"].append(prompt)
        return LLMResponse(text="prod-eu had the most P1 incidents (7).")

    monkeypatch.setattr(graph_module, "execute_sql", _execute)
    monkeypatch.setattr(graph_module, "generate_text", _generate_text)
    return env


def _pending(result: dict) -> dict:
    assert "__interrupt__" in result, "the graph should have paused for approval"
    return result["__interrupt__"][0].value


def _resume(compiled_graph, approved: bool, thread_id: str = "t1"):
    return compiled_graph.invoke(
        Command(resume=approved), {"configurable": {"thread_id": thread_id}}
    )


def test_sql_intent_pauses_for_approval_with_the_generated_select(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    result = _invoke(compiled_graph, "Which cluster had the most P1 incidents?")

    pending = _pending(result)
    assert pending["sql"].startswith("SELECT")
    assert pending["explanation"] == "Counts incidents per cluster."
    assert pending["query_id"] == "t1"
    assert sql_env["executed"] == [], "nothing may run before approval"


def test_sql_intent_does_not_run_the_rag_pipeline(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    def _boom(question: str, flags: dict) -> ChatResponse:
        raise AssertionError("a sql-intent question must not run the rag pipeline")

    monkeypatch.setattr(graph_module, "run_rag", _boom)

    _pending(_invoke(compiled_graph, "how many P1 incidents last month?"))


def test_approving_runs_the_sql_and_answers_from_the_rows(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")
    _pending(_invoke(compiled_graph, "Which cluster had the most P1 incidents?"))

    result = _resume(compiled_graph, approved=True)

    assert sql_env["executed"] == ["SELECT cluster, count(*) AS n FROM incidents GROUP BY cluster"]
    response = result["response"]
    assert response.answer == "prod-eu had the most P1 incidents (7)."
    assert response.pending_sql is None
    assert response.metadata.route == "sql"
    assert "prod-eu" in sql_env["answer_prompts"][0]
    assert "Which cluster had the most P1 incidents?" in sql_env["answer_prompts"][0]


def test_sql_route_reports_a_real_cache_hit_not_always_false(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    """Regression: `ChatResponse.cache_hit` on the SQL route used to always
    be False, even when `execute_sql` served the rows from its own cache —
    nothing surfaced that fact up to the response, so the UI's "Cache:
    hit/miss" line was silently wrong (found testing issue #34's demo UI)."""
    sql_env["cache_hit"] = True
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")
    _pending(_invoke(compiled_graph, "Which cluster had the most P1 incidents?"))

    result = _resume(compiled_graph, approved=True)

    response = result["response"]
    assert response.cache_hit is True
    assert response.metadata.cache_hit is True


def test_rejecting_ends_with_a_clear_message_and_runs_nothing(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")
    _pending(_invoke(compiled_graph, "Which cluster had the most P1 incidents?"))

    result = _resume(compiled_graph, approved=False)

    assert sql_env["executed"] == []
    assert sql_env["answer_prompts"] == []
    response = result["response"]
    assert "rejected" in response.answer.lower()
    assert response.metadata.route == "sql_rejected"
    assert response.pending_sql is None


def test_a_dangerous_generated_statement_is_refused_before_any_approval_prompt(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    def _dangerous(question: str) -> GeneratedSQL:
        raise UnsafeSQLError("Only SELECT statements are allowed")

    monkeypatch.setattr(graph_module, "generate_sql", _dangerous)

    result = _invoke(compiled_graph, "drop the incidents table")

    assert "__interrupt__" not in result
    assert result["response"].metadata.route == "sql_refused"
    assert "SELECT" in result["response"].answer
    assert result["response"].pending_sql is None
    assert sql_env["executed"] == []


def test_a_failed_generation_ends_with_a_clean_error(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    def _down(question: str) -> GeneratedSQL:
        raise SQLGenerationError("The SQL generator is unavailable")

    monkeypatch.setattr(graph_module, "generate_sql", _down)

    result = _invoke(compiled_graph, "how many incidents?")

    assert "__interrupt__" not in result
    assert result["response"].metadata.route == "sql_error"
    assert "unavailable" in result["response"].answer


def test_a_failed_execution_ends_with_a_clean_error(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")

    def _fail(sql: str) -> list[dict]:
        raise SQLExecutionError('column "nope" does not exist')

    monkeypatch.setattr(graph_module, "execute_sql", _fail)
    _pending(_invoke(compiled_graph, "how many incidents?"))

    result = _resume(compiled_graph, approved=True)

    assert result["response"].metadata.route == "sql_error"
    assert "nope" in result["response"].answer
    assert sql_env["answer_prompts"] == []


def test_hybrid_intent_synthesises_rows_and_docs_into_one_answer(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "hybrid")
    monkeypatch.setattr(
        graph_module,
        "run_rag",
        lambda question, flags: _rag_response(answer="Restart the pod. [runbook.html]"),
    )
    _pending(_invoke(compiled_graph, "list P1 incidents with remediation docs"))

    result = _resume(compiled_graph, approved=True)

    response = result["response"]
    assert response.metadata.route == "hybrid"
    assert response.answer == "prod-eu had the most P1 incidents (7)."
    assert response.sources == ["pods.html"]
    prompt = sql_env["answer_prompts"][0]
    assert "prod-eu" in prompt, "the SQL rows must reach the synthesis prompt"
    assert "Restart the pod. [runbook.html]" in prompt, "the doc answer must reach it too"


def test_hybrid_route_cache_hit_is_true_if_either_sub_call_was_cached(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    """Regression: a hybrid answer used to just inherit whichever cache_hit
    the RAG draft happened to carry, silently ignoring the SQL half's own
    cache tier entirely (found testing issue #34's demo UI)."""
    sql_env["cache_hit"] = True
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "hybrid")
    monkeypatch.setattr(
        graph_module,
        "run_rag",
        lambda question, flags: _rag_response(answer="Restart the pod. [runbook.html]"),
    )
    _pending(_invoke(compiled_graph, "list P1 incidents with remediation docs"))

    result = _resume(compiled_graph, approved=True)

    response = result["response"]
    assert response.cache_hit is True
    assert response.metadata.cache_hit is True


def test_hybrid_rejection_still_ends_with_the_rejection_message(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "hybrid")
    monkeypatch.setattr(graph_module, "run_rag", lambda question, flags: _rag_response())
    _pending(_invoke(compiled_graph, "list P1 incidents with remediation docs"))

    result = _resume(compiled_graph, approved=False)

    assert result["response"].metadata.route == "sql_rejected"


def test_the_pause_survives_between_two_separate_invocations(
    compiled_graph, monkeypatch: pytest.MonkeyPatch, sql_env: dict
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "sql")
    _pending(_invoke(compiled_graph, "how many incidents?", thread_id="ask-1"))

    snapshot = compiled_graph.get_state({"configurable": {"thread_id": "ask-1"}})

    assert snapshot.next == ("request_sql_approval",)
    assert snapshot.values["question"] == "how many incidents?"


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
