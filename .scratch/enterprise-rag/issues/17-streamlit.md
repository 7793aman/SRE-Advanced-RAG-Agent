# Module 10b — Streamlit demo UI

Type: task
Status: open
Blocked by: 15

## Question

The visual tester: auth, query with feature toggles, SQL approval flow, use-case presets,
and an eval-results dashboard.

### Deliverables
- `scripts/streamlit_app.py` — port `_reference/scripts/streamlit_app.py` (1337 lines):
  login/register against `/auth`, a query form exposing every flag, rendering of
  `ChatResponse` (answer, sources, metadata pane), the `pending_sql` → approve/reject →
  `/query/sql/execute` flow, the `USE_CASES` preset buttons, and the eval dashboard that
  reads `eval/results/*.json`.
- `Makefile` `streamlit` target (already stubbed).

### Reference
`_reference/scripts/streamlit_app.py`.

### Acceptance
- `make streamlit` → browser UI; can register, log in, ask a RAG question, see sources +
  metadata, run a SQL question and approve it, and view an eval run.
