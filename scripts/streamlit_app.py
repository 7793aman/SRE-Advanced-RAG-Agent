"""Streamlit demo UI (issue #34): auth, a query console with every retrieval
flag exposed, the SQL-approval flow, and preset example questions.

Talks to the FastAPI service over plain HTTP (`requests`), the same way any
other client would — no FastAPI internals imported here. Requires the API
already running (`make api`) at `settings.streamlit_api_base_url`.

The layout (persistent sidebar + transcript), the theme, the route-glyph
system, the inspector, and the flag-disabling rule are design decisions
recorded in spec.md's "Demo UI (issue #34)" section, made from throwaway HTML
prototypes — this file is the real implementation of that decision, not a
port of the prototype's code. One adaptation from the prototype: entries
render as Streamlit's native `st.chat_message` bubbles rather than the
prototype's flat log-entry style, and presets ask their question immediately
on click rather than only prefilling it — both are Streamlit-idiomatic
choices that keep this maintainable rather than fighting the framework for
pixel parity with a throwaway mockup.
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from app.config import settings

st.set_page_config(page_title="Query Console", page_icon="🛰️", layout="wide")

_API = settings.streamlit_api_base_url

_PRESETS = [
    "How do I debug a crashing pod?",
    "Which cluster had the most P1 incidents last month?",
    "Show P1 incidents on prod-us-east and the fix for each alert type",
]

# route -> (glyph, label). ● = single-source resolved, ◐ = hybrid (merged
# sources), ○ = unresolved/awaiting approval, ✕ = the SQL path didn't
# complete. Unknown routes fall back to a plain label in `_route_status`.
_ROUTE_LABELS: dict[str, tuple[str, str]] = {
    "rag": ("●", "documentation"),
    "hybrid": ("◐", "hybrid — docs + database"),
    "sql": ("●", "database only"),
    "sql_pending": ("○", "awaiting approval"),
    "sql_rejected": ("✕", "rejected"),
    "sql_refused": ("✕", "refused — not a safe read-only query"),
    "sql_error": ("✕", "failed"),
    "rag_general_knowledge": ("●", "general knowledge"),
}

# Only these routes actually retrieved anything — everything else (SQL-only,
# general-knowledge skip, a halted SQL path) has no relevance score or
# sources to show, and showing a "0% relevant" meter for them would be
# misleading rather than honest.
_RETRIEVAL_ROUTES = {"rag", "hybrid"}

# Flags only affect a `sql`-intent question if it never resolves purely as
# SQL (i.e. never for these two) — see the flag x intent liveness table on
# issue #34. Told to the user instead of silently disabling the sidebar,
# since the intent isn't known until the response comes back.
_ROUTE_NOTES: dict[str, str] = {
    "sql": "Answered from the database — retrieval settings weren't used for this question.",
    "rag_general_knowledge": (
        "Answered from general knowledge — no documents were searched, "
        "so retrieval settings weren't used."
    ),
}


# --- API client ------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


def api_post(path: str, body: dict[str, Any], auth: bool = True) -> dict[str, Any] | None:
    """POST to the API; on any failure, shows `st.error` and returns None so
    callers can just check for None instead of handling exceptions."""
    try:
        response = requests.post(
            f"{_API}{path}", json=body, headers=_auth_headers() if auth else {}, timeout=120
        )
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the API at {_API}: {exc}")
        return None

    if not response.ok:
        st.error(_error_detail(response))
        return None
    return response.json()  # type: ignore[no-any-return]


# --- auth --------------------------------------------------------------------


def _log_in_or_register(path: str, username: str, password: str) -> None:
    result = api_post(path, {"username": username, "password": password}, auth=False)
    if result is None:
        return
    st.session_state.token = result["token"]
    st.session_state.username = username
    st.rerun()


def render_auth_gate() -> None:
    st.title("Query Console")
    st.caption("Sign in to ask the Kubernetes ops assistant a question.")
    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab, st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in", type="primary"):
            _log_in_or_register("/auth/login", username, password)

    with register_tab, st.form("register_form"):
        username = st.text_input("Username", key="register_username")
        password = st.text_input(
            "Password", type="password", key="register_password", help="At least 8 characters."
        )
        if st.form_submit_button("Register", type="primary"):
            _log_in_or_register("/auth/register", username, password)


# --- sidebar: retrieval controls + presets ------------------------------------


def _current_flags() -> dict[str, Any]:
    return {
        "search_mode": st.session_state.search_mode,
        "enable_rerank": st.session_state.enable_rerank,
        "enable_hyde": st.session_state.enable_hyde,
        "enable_crag": st.session_state.enable_crag,
        "enable_self_reflective": st.session_state.enable_self_reflective,
        "enable_adaptive_retrieval": st.session_state.enable_adaptive_retrieval,
        "top_k": st.session_state.top_k,
    }


def render_sidebar() -> None:
    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state.username}**")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.subheader("Try a question")
        for question in _PRESETS:
            if st.button(question, key=f"preset_{question}", use_container_width=True):
                send_question(question)
                st.rerun()

        st.divider()
        st.subheader("Retrieval settings")

        # The one flag conflict knowable before a question is even sent:
        # HyDE always searches densely internally, ignoring search_mode.
        # Every other case (a question turning out to be SQL-only, adaptive
        # retrieval skipping the corpus) can't be predicted from the
        # question text, so it's disclosed on the response instead — see
        # `_ROUTE_NOTES` — rather than guessed at here.
        hyde_on = st.session_state.get("enable_hyde", False)
        st.selectbox(
            "Search mode",
            ["dense", "sparse", "hybrid"],
            key="search_mode",
            disabled=hyde_on,
            help=(
                "Ignored while HyDE is on — HyDE always searches densely."
                if hyde_on
                else "How the corpus is searched."
            ),
        )
        st.toggle("Rerank", key="enable_rerank", help="Cross-encoder re-scores the top chunks.")
        st.toggle(
            "HyDE",
            key="enable_hyde",
            help="Searches with a hypothetical answer's embedding instead of the search mode above.",
        )
        st.toggle(
            "CRAG grading",
            key="enable_crag",
            value=True,
            help="Falls back to a web search if retrieval is weak. On by default.",
        )
        st.toggle(
            "Self-RAG reflect",
            key="enable_self_reflective",
            help="Critiques and retries a weak answer once.",
        )
        st.toggle(
            "Adaptive retrieval",
            key="enable_adaptive_retrieval",
            help="Skips the corpus search entirely when the question doesn't need it.",
        )
        st.number_input("top_k", min_value=1, max_value=50, value=5, key="top_k")


# --- response rendering --------------------------------------------------------


def _clamped(score: float) -> float:
    """`st.progress` requires 0.0-1.0; nothing in the response schema
    actually guarantees a chunk score stays in that range (a reranker's raw
    score, in particular, isn't bounded), so clamp defensively rather than
    let a real answer crash the page over a display detail."""
    return max(0.0, min(1.0, score))


def _route_status(route: str) -> str:
    glyph, label = _ROUTE_LABELS.get(route, ("●", route))
    return f"{glyph} {label}"


def _resolve_sql(entry: dict[str, Any], query_id: str, approved: bool) -> None:
    result = api_post("/query/sql/execute", {"query_id": query_id, "approved": approved})
    if result is None:
        return
    entry["response"] = result
    st.rerun()


def render_pending_sql(entry: dict[str, Any], pending: dict[str, Any]) -> None:
    st.write(pending["explanation"])
    st.code(pending["sql"], language="sql")
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve & run", key=f"approve_{pending['query_id']}", type="primary"):
        _resolve_sql(entry, pending["query_id"], approved=True)
    if reject_col.button("Reject", key=f"reject_{pending['query_id']}"):
        _resolve_sql(entry, pending["query_id"], approved=False)


def render_inspector(response: dict[str, Any]) -> None:
    metadata = response["metadata"]
    with st.expander("Inspect response"):
        formatted_tab, raw_tab = st.tabs(["Formatted", "Raw JSON"])
        with formatted_tab:
            st.write(f"**Cache:** {'hit' if response['cache_hit'] else 'miss'}")
            reflection_note = f"{metadata['reflection_iterations']} iteration(s)"
            if metadata.get("reflection_score") is not None:
                reflection_note += f", score {metadata['reflection_score']:.2f}"
            st.write(f"**Reflection:** {reflection_note}")

            chunks = metadata.get("retrieved_chunks", [])
            if not chunks:
                st.caption("No chunks retrieved for this turn.")
            for chunk in chunks:
                st.progress(
                    _clamped(chunk["score"]), text=f"{chunk['source']} — {chunk['score']:.2f}"
                )
                # Retrieved chunk text is arbitrary corpus content, not
                # markdown we wrote — st.caption/st.write would parse a
                # stray "#" as a heading. st.text renders it literally.
                st.text(chunk["text"])
        with raw_tab:
            st.json(response)


def render_response(entry: dict[str, Any]) -> None:
    response = entry["response"]
    metadata = response["metadata"]
    pending = response.get("pending_sql")

    st.caption(_route_status(metadata["route"]))

    if pending:
        render_pending_sql(entry, pending)
        return

    st.write(response["answer"])

    if metadata.get("used_web_fallback"):
        st.info(
            "This answer used a web search — the documentation corpus didn't have enough to go on."
        )

    note = _ROUTE_NOTES.get(metadata["route"])
    if note:
        st.caption(note)

    if metadata["route"] in _RETRIEVAL_ROUTES:
        st.progress(
            _clamped(response["retrieval_score"]),
            text=f"relevance {response['retrieval_score']:.2f}",
        )
        if response["sources"]:
            tags = [
                f"`[{'sql' if 'sql' in source.lower() else 'doc'}]` {source}"
                for source in response["sources"]
            ]
            st.caption(", ".join(tags))

    render_inspector(response)


# --- transcript + composer -----------------------------------------------------


def send_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    response = api_post("/query", {"question": question, **_current_flags()})
    if response is None:
        return
    st.session_state.messages.append({"question": question, "response": response})


def render_transcript() -> None:
    for entry in st.session_state.messages:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            render_response(entry)


def render_composer() -> None:
    question = st.chat_input("Ask about your clusters…")
    if question:
        send_question(question)
        st.rerun()


# --- entry point ---------------------------------------------------------------


def main() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("messages", [])

    if not st.session_state.token:
        render_auth_gate()
        return

    render_sidebar()
    st.title("Query Console")
    st.caption("Ask about your clusters, or approve a generated query.")
    render_transcript()
    render_composer()


if __name__ == "__main__":
    main()
