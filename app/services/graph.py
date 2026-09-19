"""The LangGraph state machine (issues #29, #30; spec.md "Graph"): routes each
question by intent, then runs the pipeline for it. `/query` (app/api/query.py)
calls the compiled graph instead of `rag_service.run_rag` directly.

    route_intent ─┬─ rag ───────► retrieve_rag ──────────► generate_answer ─► finalize
                  ├─ sql ───────► generate_sql_node ─┐
                  └─ hybrid ────► retrieve_rag ──────┘
                                      generate_sql_node ─► request_sql_approval
                                        ─► execute_sql ─► generate_answer ─► finalize

`request_sql_approval` calls LangGraph's `interrupt()`, which checkpoints the
run and returns control to the caller with the pending SQL. A second
invocation with `Command(resume=approved)` on the same thread continues it:
approve runs the query; reject ends with a clear message. Because the pause
lives in the Postgres checkpointer, it survives the gap between two
stateless HTTP calls (user story 23). The thread id doubles as the
`query_id` the API hands back.

Every terminal failure on the SQL path (refused statement, generator or
database failure, rejection) sets `halted` and a ready-made `response`, and
the conditional edges short-circuit to `finalize`. On `hybrid` that message
replaces the RAG answer that was already computed — a half answer that
silently omits the data the user asked for would be misleading.

Node granularity note: `rag_service.run_rag` already owns retrieval,
generation, CRAG, the Self-RAG reflection loop, and the rag_answer cache as
one atomic, already-tested unit — the reflection loop re-retrieves against
a refined question, so retrieval and generation can't be safely split into
two independent, non-looping graph nodes without reimplementing that loop's
control flow at the graph level. Rather than risk regressing #28's tested
behaviour, `retrieve_rag` calls `run_rag` unchanged. `generate_answer` is a
pass-through on the rag path and, when SQL rows are present, synthesises them
(plus the RAG draft, for `hybrid`) into the final answer.

`build_graph(checkpointer)` takes its checkpointer as a parameter so tests
can compile against `MemorySaver` with no Postgres dependency. `get_graph()`
is the production singleton: compiled lazily, against a live Postgres
checkpointer, on first call — not at import time — mirroring
`llm_service._get_client()`'s lazy-singleton pattern, so importing this
module (or `app.main`) never requires a reachable Postgres just to run
unrelated tests.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.config import settings
from app.models import ChatResponse, ResponseMetadata
from app.security.system_prompt import SQL_ANSWER_SYSTEM_PROMPT
from app.services.llm_service import generate_text
from app.services.rag_service import run_rag
from app.services.router_service import classify_intent
from app.services.sql_service import (
    SQLExecutionError,
    SQLGenerationError,
    UnsafeSQLError,
    execute_sql,
    generate_sql,
)

_REFUSED_ANSWER = "I can only run read-only SELECT queries, so I didn't run that one. ({reason})"
_REJECTED_ANSWER = "The SQL query was rejected, so nothing was run against the database."

# Holds `get_graph()`'s checkpointer context manager alive for the process's
# lifetime — see the warning in `get_graph()`'s docstring.
_checkpointer_cm: object | None = None


class GraphState(TypedDict, total=False):
    question: str
    flags: dict[str, Any]
    user_id: int
    intent: str
    sql: str
    sql_explanation: str
    rows: list[dict[str, Any]]
    # Set by any SQL-path node that ends the run early with its own `response`.
    halted: bool
    response: ChatResponse


def _halt(answer: str, route: str) -> dict[str, Any]:
    response = ChatResponse(
        answer=answer, sources=[], retrieval_score=0.0, metadata=ResponseMetadata(route=route)
    )
    return {"response": response, "halted": True}


def route_intent(state: GraphState) -> dict[str, Any]:
    return {"intent": classify_intent(state["question"])}


def _after_route(state: GraphState) -> str:
    return "generate_sql_node" if state["intent"] == "sql" else "retrieve_rag"


def retrieve_rag(state: GraphState) -> dict[str, Any]:
    return {"response": run_rag(state["question"], state["flags"])}


def _after_retrieve(state: GraphState) -> str:
    return "generate_sql_node" if state["intent"] == "hybrid" else "generate_answer"


def generate_sql_node(state: GraphState) -> dict[str, Any]:
    try:
        generated = generate_sql(state["question"])
    except UnsafeSQLError as exc:
        return _halt(_REFUSED_ANSWER.format(reason=exc), "sql_refused")
    except SQLGenerationError as exc:
        return _halt(f"I couldn't generate a SQL query for that: {exc}.", "sql_error")
    return {"sql": generated.sql, "sql_explanation": generated.explanation}


def request_sql_approval(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    # Nothing else may run in this node: on resume LangGraph re-executes it
    # from the top, and `interrupt()` then returns the resume value.
    approved = interrupt(
        {
            "sql": state["sql"],
            "explanation": state.get("sql_explanation", ""),
            "query_id": config["configurable"]["thread_id"],
        }
    )
    if approved is True:
        return {}
    return _halt(_REJECTED_ANSWER, "sql_rejected")


def execute_sql_node(state: GraphState) -> dict[str, Any]:
    try:
        return {"rows": execute_sql(state["sql"])}
    except (SQLExecutionError, UnsafeSQLError) as exc:
        return _halt(f"The approved query couldn't be run: {exc}", "sql_error")


def _continue_unless_halted(next_node: str):  # noqa: ANN202
    def _edge(state: GraphState) -> str:
        return "finalize" if state.get("halted") else next_node

    return _edge


def _synthesis_prompt(state: GraphState) -> str:
    parts = [
        f"Question: {state['question']}",
        f"SQL that was run: {state['sql']}",
        f"<sql_rows>\n{json.dumps(state['rows'], default=str)}\n</sql_rows>",
    ]
    if state["intent"] == "hybrid":
        parts.append(f"<documentation_answer>\n{state['response'].answer}\n</documentation_answer>")
    return "\n\n".join(parts)


def generate_answer(state: GraphState) -> dict[str, Any]:
    if "rows" not in state:
        # rag path: generation already happened inside `retrieve_rag`.
        return {}

    answer = generate_text(_synthesis_prompt(state), system_prompt=SQL_ANSWER_SYSTEM_PROMPT).text
    if state["intent"] == "hybrid":
        rag = state["response"]
        response = rag.model_copy(
            update={
                "answer": answer,
                "metadata": rag.metadata.model_copy(update={"route": "hybrid"}),
            }
        )
    else:
        response = ChatResponse(
            answer=answer, sources=[], retrieval_score=0.0, metadata=ResponseMetadata(route="sql")
        )
    return {"response": response}


def finalize(state: GraphState) -> dict[str, Any]:
    return {"response": state["response"]}


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    builder = StateGraph(GraphState)
    builder.add_node("route_intent", route_intent)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("generate_sql_node", generate_sql_node)
    builder.add_node("request_sql_approval", request_sql_approval)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        _after_route,
        {"generate_sql_node": "generate_sql_node", "retrieve_rag": "retrieve_rag"},
    )
    builder.add_conditional_edges(
        "retrieve_rag",
        _after_retrieve,
        {"generate_sql_node": "generate_sql_node", "generate_answer": "generate_answer"},
    )
    builder.add_conditional_edges(
        "generate_sql_node",
        _continue_unless_halted("request_sql_approval"),
        {"request_sql_approval": "request_sql_approval", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "request_sql_approval",
        _continue_unless_halted("execute_sql"),
        {"execute_sql": "execute_sql", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "execute_sql",
        _continue_unless_halted("generate_answer"),
        {"generate_answer": "generate_answer", "finalize": "finalize"},
    )
    builder.add_edge("generate_answer", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """The production singleton: compiled once, against a live Postgres
    checkpointer, the first time it's called. Never opened per-request —
    it's a process-lifetime singleton, same as `llm_service._get_client()`'s
    OpenAI client.

    `from_conn_string` is a `@contextmanager`, not a plain constructor: its
    connection lives inside that generator's frame, so the context-manager
    object itself must stay referenced for the connection's lifetime — a
    bare `.from_conn_string(...).__enter__()` lets the generator get
    garbage-collected immediately, which closes the connection out from
    under the checkpointer on its very first real use.

    `lru_cache` doesn't cache exceptions, so a `checkpointer.setup()`
    failure (e.g. a transient DB blip) leaves this function free to run
    again on the next call. The module-level reference is only assigned
    *after* `setup()` succeeds, and a failed attempt explicitly closes its
    own connection first — otherwise every retry during an outage would
    open one more live connection with nothing left to ever close it,
    leaking connections until the pool is exhausted.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    checkpointer_cm = PostgresSaver.from_conn_string(settings.database_url)
    checkpointer = checkpointer_cm.__enter__()
    try:
        checkpointer.setup()
    except BaseException:
        checkpointer_cm.__exit__(*sys.exc_info())
        raise

    global _checkpointer_cm
    _checkpointer_cm = checkpointer_cm
    return build_graph(checkpointer)
