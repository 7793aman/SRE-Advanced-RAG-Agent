# Module 1 — Native RAG pipeline + /query + /admin

Type: task
Status: open
Blocked by: 03, 04, 05

## Question

Wire the naïve dense-retrieval RAG path end to end and expose it. This completes the
baseline whose weakness (drowning in 95% noise) motivates every later module.

### Deliverables
- `app/services/llm_service.py` — `generate(system, user, model=None, temperature=0.0)` and
  `generate_with_json(...)` (OpenAI chat completions; return `{"text", "usage"}`).
- `app/security/spotlighting.py` — `build_spotlighted_context(chunks)` (XML `<retrieved_context>`
  wrap + "UNTRUSTED DATA" preamble). Needed by rag_service from day one (it's L8, but the
  reference calls it at baseline).
- `app/security/system_prompt.py` — `build_system_prompt()` returning the hardened prompt.
- `app/services/rag_service.py` — `_retrieve` (dense only for now: embed → `search`;
  `top_k` from flags), `_generate` (spotlight → system prompt → `generate`), `run_rag` (with
  rag_answer cache lookup/write), `run_rag_with_trace` / `run_rag_with_trace_no_cache` (for eval).
- `app/api/query.py` — `POST /query` (JWT-protected via `get_current_user`): build `flags`
  dict, call `run_rag`, return `ChatResponse`. Per-user rate limit check.
- `app/api/admin.py` — `GET /admin/health` (ping postgres/qdrant/redis/openai/tavily),
  `GET /admin/cache/stats` (admin), `POST /admin/cache/clear` (admin).
- Wire all three routers into `app/main.py`.
- Full ingestion: `make seed` (with ticket-04's noise sample) populates Qdrant.

### Reference
commit `1d9e264` — `app/services/{llm_service,rag_service}.py`, `app/api/{query,admin}.py`,
`app/security/{spotlighting,system_prompt}.py`. Note: `rag_service.py` at HEAD imports many
later modules — build the **baseline** version (dense retrieve + generate only).

### Acceptance
- `curl -H "Authorization: Bearer <jwt>" -d '{"question":"What is a Kubernetes Deployment?"}' /query`
  returns a grounded answer citing a `true_data` source.
- A deliberately hard query (short, lexical, e.g. "imagePullPolicy Always") visibly retrieves
  noise — the motivating failure. Note it in the resolution.
- `GET /admin/health` → all green.
