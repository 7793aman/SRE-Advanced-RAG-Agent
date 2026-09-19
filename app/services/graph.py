"""The LangGraph state machine (issue #29; spec.md "Graph"): routes each
question by intent, then runs the pipeline for it. `/query` (app/api/query.py)
calls the compiled graph instead of `rag_service.run_rag` directly.

Only the `rag` path is wired up today. `sql` and `hybrid` classify
correctly (see `router_service.py`) but end at an explicit "not supported
yet" response in `finalize` rather than silently running the rag pipeline
under the wrong label — Text2SQL, SQL approval, and hybrid synthesis are a
later ticket (spec.md's fuller node list adds `generate_sql_node`,
`request_sql_approval`, `execute_sql`).

Node granularity note: `rag_service.run_rag` already owns retrieval,
generation, CRAG, the Self-RAG reflection loop, and the rag_answer cache as
one atomic, already-tested unit — the reflection loop re-retrieves against
a refined question, so retrieval and generation can't be safely split into
two independent, non-looping graph nodes without reimplementing that loop's
control flow at the graph level. Rather than risk regressing #28's tested
behaviour, `retrieve_rag` calls `run_rag` unchanged; `generate_answer` is
kept as its own node (as the ticket and spec.md both name it) but is
presently a pass-through — the seam a later ticket can use to synthesize
RAG + SQL evidence for `hybrid`.

`build_graph(checkpointer)` takes its checkpointer as a parameter so tests
can compile against `MemorySaver` with no Postgres dependency. `get_graph()`
is the production singleton: compiled lazily, against a live Postgres
checkpointer, on first call — not at import time — mirroring
`llm_service._get_client()`'s lazy-singleton pattern, so importing this
module (or `app.main`) never requires a reachable Postgres just to run
unrelated tests.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.models import ChatResponse, ResponseMetadata
from app.services.rag_service import run_rag
from app.services.router_service import classify_intent

_UNSUPPORTED_ANSWER = (
    "This question needs {intent} evidence, which this system doesn't answer yet."
)

# Holds `get_graph()`'s checkpointer context manager alive for the process's
# lifetime — see the warning in `get_graph()`'s docstring.
_checkpointer_cm: object | None = None


class GraphState(TypedDict, total=False):
    question: str
    flags: dict[str, Any]
    intent: str
    response: ChatResponse


def route_intent(state: GraphState) -> dict[str, Any]:
    return {"intent": classify_intent(state["question"])}


def _select_branch(state: GraphState) -> str:
    return "rag" if state["intent"] == "rag" else "unsupported"


def retrieve_rag(state: GraphState) -> dict[str, Any]:
    return {"response": run_rag(state["question"], state["flags"])}


def generate_answer(state: GraphState) -> dict[str, Any]:
    # See module docstring: generation already happened inside `retrieve_rag`.
    # Nothing to do here yet for the rag path — this seam is reserved for
    # future hybrid RAG+SQL answer synthesis.
    return {}


def _unsupported_response(intent: str) -> ChatResponse:
    return ChatResponse(
        answer=_UNSUPPORTED_ANSWER.format(intent=intent),
        sources=[],
        retrieval_score=0.0,
        metadata=ResponseMetadata(route=f"{intent}_not_supported"),
    )


def finalize(state: GraphState) -> dict[str, Any]:
    if "response" in state:
        return {"response": state["response"]}
    return {"response": _unsupported_response(state["intent"])}


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    builder = StateGraph(GraphState)
    builder.add_node("route_intent", route_intent)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        _select_branch,
        {"rag": "retrieve_rag", "unsupported": "finalize"},
    )
    builder.add_edge("retrieve_rag", "generate_answer")
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
