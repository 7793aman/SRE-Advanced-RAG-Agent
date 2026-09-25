"""Unit seam: `get_graph()`'s lazy-singleton and connection lifecycle, with
`PostgresSaver.from_conn_string` faked. Nothing here touches a real
Postgres connection.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.services import graph as graph_module


class _FakeCheckpointerCM:
    """Stands in for `PostgresSaver.from_conn_string(...)`'s context manager.
    `__enter__` returns a real `BaseCheckpointSaver` (a `MemorySaver`, so
    `build_graph`'s type check accepts it) whose `.setup()` can be made to
    fail on demand."""

    def __init__(self, fail_setup: bool) -> None:
        self.exited = False
        self.checkpointer = MemorySaver()

        def _setup() -> None:
            if fail_setup:
                raise RuntimeError("Postgres is momentarily unreachable")

        self.checkpointer.setup = _setup  # type: ignore[method-assign]

    def __enter__(self) -> MemorySaver:
        return self.checkpointer

    def __exit__(self, *exc_info: object) -> bool:
        self.exited = True
        return False


@pytest.fixture(autouse=True)
def _reset_graph_singleton() -> Iterator[None]:
    graph_module._graph_singleton.reset()
    graph_module._checkpointer_cm = None
    yield
    graph_module._graph_singleton.reset()
    graph_module._checkpointer_cm = None


def test_get_graph_is_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string",
        lambda conn_string: _FakeCheckpointerCM(fail_setup=False),
    )

    assert graph_module.get_graph() is graph_module.get_graph()


def test_a_failed_setup_closes_the_connection_instead_of_leaking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_cm = _FakeCheckpointerCM(fail_setup=True)
    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string",
        lambda conn_string: failing_cm,
    )

    with pytest.raises(RuntimeError, match="momentarily unreachable"):
        graph_module.get_graph()

    assert failing_cm.exited is True
    assert graph_module._checkpointer_cm is None


def test_retry_after_a_failed_setup_succeeds_and_does_not_touch_the_failed_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_cm = _FakeCheckpointerCM(fail_setup=True)
    working_cm = _FakeCheckpointerCM(fail_setup=False)
    attempts = iter([failing_cm, working_cm])

    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string",
        lambda conn_string: next(attempts),
    )

    with pytest.raises(RuntimeError):
        graph_module.get_graph()

    compiled = graph_module.get_graph()

    assert compiled is not None
    assert failing_cm.exited is True
    assert working_cm.exited is False  # the live singleton's connection stays open
