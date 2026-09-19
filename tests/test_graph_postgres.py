"""Integration seam: `get_graph()` against a *live* Postgres checkpointer —
issue #29's first "Done when": "graph compiles ... with a live Postgres
checkpointer". Skips if Postgres isn't reachable, same convention as the
rest of the suite's live-Postgres tests (see `db_ready` in conftest.py).
`router_service.classify_intent` and `rag_service.run_rag` are faked; no
OpenAI or Qdrant calls here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.models import ChatResponse, ResponseMetadata
from app.services import graph as graph_module


@pytest.fixture(autouse=True)
def _clear_graph_cache() -> Iterator[None]:
    graph_module.get_graph.cache_clear()
    yield
    graph_module.get_graph.cache_clear()


def test_get_graph_compiles_with_a_live_postgres_checkpointer(db_ready: None) -> None:
    compiled = graph_module.get_graph()

    assert compiled is graph_module.get_graph(), "get_graph() must be a singleton, not rebuilt"


def test_a_doc_question_routes_rag_and_its_state_persists_in_postgres(
    db_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "classify_intent", lambda question: "rag")
    monkeypatch.setattr(
        graph_module,
        "run_rag",
        lambda question, flags: ChatResponse(
            answer="A Pod is the smallest deployable unit. [pods.html]",
            sources=["pods.html"],
            retrieval_score=0.9,
            metadata=ResponseMetadata(route="rag"),
        ),
    )

    compiled = graph_module.get_graph()
    config = {"configurable": {"thread_id": "test-thread-doc-question"}}
    result = compiled.invoke({"question": "What is a Pod?", "flags": {}}, config)

    assert result["response"].answer == "A Pod is the smallest deployable unit. [pods.html]"

    # The whole point of a Postgres checkpointer: a fresh read of the thread's
    # state comes back from Postgres, not just from the dict `invoke` returned.
    persisted = compiled.get_state(config)
    assert persisted.values["intent"] == "rag"
    assert persisted.values["question"] == "What is a Pod?"
