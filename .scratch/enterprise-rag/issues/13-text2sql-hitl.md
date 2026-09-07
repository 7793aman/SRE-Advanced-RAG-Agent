# Module 7b — Text2SQL with human-in-the-loop approval

Type: task
Status: open
Blocked by: 12

## Question

Generate SQL from natural language, pause the graph for human approval via `interrupt()`,
resume on approve/reject, and answer from the rows.

### Deliverables
- `app/services/sql_service.py` — `SQLService`: `_build_schema_context` (introspect
  `information_schema.columns`), `generate_sql(question)` (LLM → `{sql, explanation}`,
  cached via `sql_gen` tier), `is_select_only(sql)` (starts with select + forbidden-keyword
  blocklist), `execute_sql(sql)` (SELECT-only guard, `sql_result` cache, row serialization).
- `app/core/graph.py` — nodes `generate_sql_node`, `request_sql_approval` (`interrupt({...})`),
  `execute_sql`; edges `retrieve_rag → generate_sql_node` (hybrid), `generate_sql_node →
  request_sql_approval → execute_sql → generate_answer`. `generate_answer` handles `sql` and
  `hybrid` intents (`_generate_hybrid_answer` synthesizes rows + rag context).
- `app/api/query.py` — detect `__interrupt__` in the invoke result → return `ChatResponse`
  with `pending_sql`; `POST /query/sql/execute` → `graph.invoke(Command(resume={"approved":
  ...}), config={thread_id})`.

### Reference
commit `d27a1d0` — `app/services/sql_service.py`, `app/core/graph.py`, `app/api/query.py`,
`app/models.py` (`PendingSQLBlock`), `app/services/rag_service.py` (`_run_sql_inline`,
`_run_hybrid_inline`).

### Acceptance
- "Which cluster had the most P1 incidents?" → response has `pending_sql` with a SELECT.
- `POST /query/sql/execute {approved:true}` → rows → natural-language answer.
- `{approved:false}` → "SQL query was not approved."
- An injected `DROP TABLE` in generated SQL is rejected by `is_select_only`.
- Hybrid question returns a synthesized answer using both rows and docs.
