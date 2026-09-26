"""Regression coverage for `scripts/streamlit_app.py`'s flag wiring.

Found the hard way (issue #34 follow-up): the sidebar's preset-question
buttons sit *above* the retrieval-settings toggles in `render_sidebar`, and
each preset's click handler calls `st.rerun()` immediately. That aborts the
script before the toggles below render again this pass, and Streamlit resets
their session_state for not having been re-registered in the run that just
ended — so a preset click could silently send every retrieval flag (HyDE,
rerank, search mode, ...) at its *default*, regardless of what the sidebar
visibly showed. Typing a question into the composer never triggered it,
since the whole sidebar (composer sits after it in `main`) had already
rendered by the time its own `st.rerun()` fired.

The fix (`send_question` snapshots `_current_flags()` at click time, before
any rerun can truncate anything, and `_fetch_response` uses that snapshot
instead of re-reading session_state later) is covered here across every
entry point (each preset button, the composer) crossed with every flag this
matters for, using Streamlit's own headless `AppTest` — no browser, no live
API; `requests.post` is stubbed and every captured `/query` body is asserted
against what the sidebar was actually set to at send time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

_APP_PATH = str(Path(__file__).resolve().parents[1] / "scripts" / "streamlit_app.py")

_STUB_METADATA = {
    "route": "rag",
    "retrieved_chunks": [],
    "cache_hit": False,
    "reflection_iterations": 0,
    "reflection_score": None,
    "refined_question": None,
    "used_web_fallback": False,
    "retrieval_path": "stub",
}

_STUB_RESPONSE = {
    "answer": "stub answer",
    "sources": [],
    "retrieval_score": 0.0,
    "pending_sql": None,
    "cache_hit": False,
    "metadata": _STUB_METADATA,
}


def _fake_post_capturing(bodies: list[dict[str, Any]]):
    def _fake_post(url: str, json: dict[str, Any] | None = None, **_: Any) -> MagicMock:
        if url.endswith("/query"):
            bodies.append(json or {})
        response = MagicMock()
        response.status_code = 200
        response.ok = True
        response.json.return_value = _STUB_RESPONSE
        return response

    return _fake_post


@pytest.fixture
def logged_in_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.session_state["token"] = "fake-token"
    at.session_state["username"] = "tester"
    at.run()
    return at


_DEFAULT_FLAGS = {
    "search_mode": "dense",
    "enable_rerank": False,
    "enable_hyde": False,
    "enable_crag": True,
    "enable_self_reflective": False,
    "enable_adaptive_retrieval": False,
    "top_k": 5,
}


def _toggle(app: AppTest, key: str, value: bool) -> None:
    widget = next(t for t in app.toggle if t.key == key)
    widget.set_value(value).run()


@pytest.mark.parametrize(
    "toggle_key",
    ["enable_hyde", "enable_rerank", "enable_self_reflective", "enable_adaptive_retrieval"],
)
def test_preset_click_sends_current_flag_state(logged_in_app: AppTest, toggle_key: str) -> None:
    """Each retrieval flag, toggled on, must reach the backend as on when a
    preset question is clicked — not silently reset to its default."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        _toggle(logged_in_app, toggle_key, True)

        preset = logged_in_app.button[0]
        preset.click().run()
        logged_in_app.run()  # process the queued fetch

    assert bodies, "no /query request was captured"
    assert bodies[-1][toggle_key] is True


def test_preset_click_with_every_flag_on(logged_in_app: AppTest) -> None:
    """Every boolean flag flipped at once, still a preset click — nothing
    should fall back to its default."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        for key in (
            "enable_rerank",
            "enable_hyde",
            "enable_self_reflective",
            "enable_adaptive_retrieval",
        ):
            _toggle(logged_in_app, key, True)
        # CRAG defaults on; flipping it off is the interesting case for it.
        _toggle(logged_in_app, "enable_crag", False)

        preset = logged_in_app.button[0]
        preset.click().run()
        logged_in_app.run()

    assert bodies
    sent = bodies[-1]
    assert sent["enable_rerank"] is True
    assert sent["enable_hyde"] is True
    assert sent["enable_self_reflective"] is True
    assert sent["enable_adaptive_retrieval"] is True
    assert sent["enable_crag"] is False


def test_preset_click_sends_selected_search_mode(logged_in_app: AppTest) -> None:
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        selectbox = next(s for s in logged_in_app.selectbox if s.key == "search_mode")
        selectbox.set_value("hybrid").run()

        preset = logged_in_app.button[0]
        preset.click().run()
        logged_in_app.run()

    assert bodies
    assert bodies[-1]["search_mode"] == "hybrid"


def test_preset_click_sends_custom_top_k(logged_in_app: AppTest) -> None:
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        number_input = next(n for n in logged_in_app.number_input if n.key == "top_k")
        number_input.set_value(17).run()

        preset = logged_in_app.button[0]
        preset.click().run()
        logged_in_app.run()

    assert bodies
    assert bodies[-1]["top_k"] == 17


def test_preset_click_with_no_flags_touched_sends_defaults(logged_in_app: AppTest) -> None:
    """Sanity check on the defaults themselves, so the other cases are
    testing an actual change and not just an accidentally-matching default."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        preset = logged_in_app.button[0]
        preset.click().run()
        logged_in_app.run()

    assert bodies
    assert bodies[-1] == {"question": preset.label, **_DEFAULT_FLAGS}


@pytest.mark.parametrize("preset_index", [0, 1, 2])
def test_every_preset_button_sends_current_hyde_state(
    logged_in_app: AppTest, preset_index: int
) -> None:
    """Not just the first preset — all three buttons sit above the toggles
    in render order, so all three are equally exposed to the truncation
    bug."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        _toggle(logged_in_app, "enable_hyde", True)

        preset = logged_in_app.button[preset_index]
        preset.click().run()
        logged_in_app.run()

    assert bodies
    assert bodies[-1]["enable_hyde"] is True


def test_typed_question_sends_current_flag_state(logged_in_app: AppTest) -> None:
    """The composer sits after the whole sidebar in `main`, so it was never
    exposed to this bug — kept as a regression guard against reintroducing
    it there too."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        _toggle(logged_in_app, "enable_hyde", True)
        logged_in_app.chat_input[0].set_value("A typed question").run()
        logged_in_app.run()

    assert bodies
    assert bodies[-1]["enable_hyde"] is True


def test_flags_snapshot_is_per_message_not_re_read_at_fetch_time(logged_in_app: AppTest) -> None:
    """Queue one question with HyDE on, then flip HyDE off before the fetch
    would run: the request already in flight must keep using the flags that
    were active when it was actually asked, not whatever the sidebar has
    drifted to by fetch time."""
    bodies: list[dict[str, Any]] = []
    with patch("requests.post", side_effect=_fake_post_capturing(bodies)):
        _toggle(logged_in_app, "enable_hyde", True)

        preset = logged_in_app.button[0]
        # This click's own `st.rerun()` fires and aborts the script right
        # here — the question is queued with its flags snapshot, but the
        # fetch itself hasn't happened yet (that's `render_transcript`,
        # further down `main`, which this aborted pass never reached).
        preset.click().run()

        # Flip the toggle back off *before* the queued fetch ever runs.
        # `_toggle`'s own `.run()` is a full, uninterrupted pass this time
        # (no button was clicked to abort it early), so it's the one that
        # reaches `render_transcript` and actually fires the fetch.
        _toggle(logged_in_app, "enable_hyde", False)

    assert bodies
    assert bodies[-1]["enable_hyde"] is True
