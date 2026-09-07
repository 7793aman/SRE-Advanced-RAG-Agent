# Module 7a — LangGraph state machine + intent router

Type: task
Status: open
Blocked by: 11, 04

## Question

Replace the inline `run_rag` call in `/query` with a compiled LangGraph state machine,
and add the intent router that decides rag / sql / hybrid.

### Deliverables
- `app/services/router_service.py` — `classify_intent(question) -> "sql"|"rag"|"hybrid"`
  (LLM JSON classifier, cached via `query_cache.get_intent/set_intent`). Decide what to do
  with the leftover `_DOCUMENT_HINTS` hack — faithfully copy or drop (document the call).
- `app/core/state.py` — `GraphState` TypedDict (question, user_id, flags, intent, sql_*,
  hypotheses, retrieved_chunks (Annotated add-reducer), reflection_*, final_answer, sources,
  confidence, cache_hits, ...).
- `app/core/graph.py` — nodes `route_intent`, `retrieve_rag`, `generate_answer`, `finalize`;
  `START → route_intent`; conditional edges on `intent` (`rag → generate_answer`,
  `hybrid → retrieve_rag`, `sql → generate_sql_node` [stub until ticket 13]); Postgres
  checkpointer via `langgraph.checkpoint.postgres.PostgresSaver` (`.setup()`).
  `generate_answer` for the rag branch calls `run_rag`.
- `app/api/query.py` — `graph.invoke({...}, config={"configurable":{"thread_id": uuid}})`;
  handle a plain rag result. (SQL interrupt handling → ticket 13.)

### Reference
commit `d27a1d0` — `app/core/graph.py`, `app/core/state.py`, `app/services/router_service.py`,
`app/api/query.py` diff, `app/models.py` diff.

### Acceptance
- `graph` compiles at import with a live Postgres checkpointer.
- A doc question routes `rag` and still returns the same grounded answer as ticket 11.
- `classify_intent` returns `sql` for "how many P1 incidents last month".
