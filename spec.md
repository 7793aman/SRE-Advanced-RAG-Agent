# Spec — Enterprise RAG for Kubernetes IT-Operations

Status: ready-for-agent

## Problem Statement

Site-reliability and platform engineers need fast, trustworthy answers to two very
different kinds of question during operational work:

- **Conceptual / procedural** — "How does a Deployment do a rolling update?", "Walk me
  through debugging CrashLoopBackOff." The answer lives in documentation and runbooks.
- **Operational / factual** — "Which cluster had the most P1 incidents last month?",
  "What's our average MTTR for network alerts?" The answer lives in a database.

A naïve retrieval-augmented chatbot (embed the question, pull the top-k most similar
document chunks, stuff them in the prompt, generate) fails them in practice:

- The knowledge base is mostly irrelevant material, so top-k retrieval returns noise and
  the model answers from noise or hallucinates.
- Short operational queries share little vocabulary with long documentation prose, so the
  right document isn't retrieved at all.
- It cannot answer questions whose answer is a number in a database.
- It has no defence against prompt injection, leaks PII, and has unbounded latency and
  cost under load.

## Solution

A single FastAPI service — an operations copilot — that an SRE can actually rely on:

- An **intent router** classifies each question as `rag`, `sql`, or `hybrid` and sends it
  down the right path.
- The **RAG path** layers every advanced retrieval technique that earns its place against
  a deliberately noisy corpus: hybrid dense+sparse search fused with Reciprocal Rank
  Fusion, cross-encoder reranking, HyDE hypothetical-answer retrieval, CRAG relevance
  grading with web-search fallback, and a Self-RAG reflection loop that regenerates weak
  answers and skips retrieval when it isn't needed.
- The **SQL path** turns the question into a PostgreSQL `SELECT`, **pauses for the human
  to approve the exact query**, then runs it and answers from the rows.
- The **hybrid path** runs both and synthesises one answer.
- Everything is orchestrated as a **stateful, resumable LangGraph state machine** with a
  Postgres checkpointer, so the SQL-approval pause survives across HTTP requests.
- Every request passes through a **fixed-order security pipeline** (input validation, JWT
  auth, rate limiting, token budgets, input restructuring, prompt-injection scanning, PII
  redaction on input and output, a hardened system prompt, spotlighting of retrieved
  chunks, and structured-output validation with LLM retry).
- Every expensive call (embedding, intent classification, SQL generation, SQL execution,
  full answer) is wrapped in a **content-hash-keyed cache** with per-workload TTLs, and
  cache hits are reported in the response.
- A **Ragas eval harness** scores retrieval and answer quality on a fixed question set and
  proves each technique beats the naïve baseline.
- A **Streamlit UI** exercises auth, querying with all feature toggles, the SQL-approval
  flow, and the eval dashboard.
- The whole stack runs from `docker compose up`.

The knowledge base is intentionally **95% noise / 5% signal** so every advanced technique
must demonstrably rescue the signal — this is also what makes the project a good teaching
artefact.

## User Stories

### Asking questions

1. As an SRE, I want to POST a natural-language question to `/query` and get a grounded
   answer with cited sources, so that I can trust and verify it.
2. As an SRE, I want the system to decide by itself whether my question needs
   documentation, database data, or both, so that I don't have to pick a mode.
3. As an SRE, I want a conceptual question ("what is a StatefulSet?") answered from the
   Kubernetes docs with the source filenames listed.
4. As an SRE, I want a factual question ("how many P1 incidents last month?") answered
   from the operational database, not from documentation.
5. As an SRE, I want a combined question ("show P1 incidents on prod-us-east and the
   remediation steps for each alert type") answered from both sources in one response.
6. As an SRE, I want per-request toggles for each advanced technique (`search_mode`,
   `enable_rerank`, `enable_hyde`, `enable_crag`, `enable_self_reflective`, `top_k`), so
   that I can compare behaviours and the eval harness can profile them.
7. As an SRE, I want the response to tell me which retrieval path ran, which chunks were
   used, the reranker/CRAG signals, and whether it was a cache hit, so that I can debug
   answer quality.

### Retrieval quality (the advanced techniques)

8. As an SRE, I want dense semantic search so that "progressive rollout" finds docs about
   "rolling updates".
9. As an SRE, I want sparse lexical search so that exact tokens like `imagePullPolicy` and
   `CrashLoopBackOff` reliably retrieve the right doc.
10. As an SRE, I want dense and sparse results fused with Reciprocal Rank Fusion so that
    hybrid retrieval is at least as good as the better of the two. **Bug found during
    issue #34 (Streamlit demo UI) prototyping, fixed as part of that same ticket:**
    `hybrid_search`'s per-side candidate overfetch before fusion was implemented but
    dead-commented (`app/services/vector_store.py`), so each side was only fetching
    `top_k` candidates instead of a wider pool — a chunk ranked low in one side but high
    in the other could be dropped before RRF ever saw it, undermining this story's own
    acceptance criterion. Re-enable the overfetch multiplier.
11. As an SRE, I want a cross-encoder to rerank the retrieved candidates so that the
    genuinely relevant chunk is promoted above superficially-similar noise.
12. As an SRE, I want the reranker backend to be pluggable (a local model or a hosted API)
    via configuration.
13. As an SRE, I want HyDE to retrieve using LLM-generated hypothetical answers so that my
    short symptom-style query ("why is my pod OOMKilled?") still finds the right doc.
14. As an SRE, I want CRAG to grade whether the retrieved chunks are actually relevant, so
    that a weak retrieval doesn't become a confident wrong answer.
15. As an SRE, I want CRAG to fall back to web search when the corpus can't answer (e.g.
    "what's the latest stable Kubernetes release?"), so that I get a useful answer with a
    web source instead of a hallucination. **Gap found during issue #34 (Streamlit demo
    UI) prototyping, fixed as part of that same ticket:** `ChatResponse.metadata.route`
    stays `"rag"` even when CRAG corrected with a web search — the demo UI needs a
    trustworthy way to tell the user "this came from the web, not our docs," and today
    the only signal is inferring it from `sources` (URLs vs. `.md` filenames), which the
    UI shouldn't have to do. Add an explicit signal (exact shape — a new `route` value
    or a boolean field — still to be decided alongside the rest of the UI design).
16. As an SRE, I want a Self-RAG reflection loop to critique the generated answer and
    regenerate with a sharpened question if it's weak, up to a retry limit.
17. As an SRE, I want Self-RAG to skip retrieval entirely for general knowledge questions
    it can answer directly.
18. As an operator of the system, I want retrieval to survive a missing optional
    dependency or API key by degrading gracefully rather than crashing.

### Text2SQL with human-in-the-loop

19. As an SRE, I want my data question turned into a PostgreSQL `SELECT` against the
    operational schema.
20. As an SRE, I want to see the exact generated SQL and its explanation and explicitly
    approve or reject it before anything runs against the database.
21. As an SRE, I want a rejected query to end gracefully with a clear message.
22. As an SRE, I want the system to refuse to execute anything that isn't a plain `SELECT`
    (no `INSERT`/`UPDATE`/`DELETE`/`DROP`/etc.), so that Text2SQL can't damage data.
23. As an SRE, I want the approval pause to survive the gap between two separate HTTP calls
    (ask, then approve), so that the flow works over a stateless API.
24. As an SRE, I want the SQL schema discovered from the live database, so that the
    generator always matches the real tables.

### The state machine

25. As an operator, I want the whole flow modelled as an explicit LangGraph state machine
    with named nodes and conditional edges on intent, so that the control flow is legible
    and resumable.
26. As an operator, I want graph state persisted in Postgres (a checkpointer), so that an
    interrupted run can be resumed by thread id.

### Security

27. As an operator, I want obvious prompt-injection and script-injection patterns rejected
    at the schema-validation layer before any processing.
28. As an operator, I want every `/query` call to require a valid JWT.
29. As an operator, I want per-user sliding-window rate limiting so that one user can't
    flood the service.
30. As an operator, I want a per-user daily token budget so that LLM spend is bounded, and
    a clear error when a user is over budget.
31. As an operator, I want over-long input truncated or summarised to a token ceiling
    before it reaches the model.
32. As an operator, I want a dedicated prompt-injection / toxicity / banned-topic scan on
    the input beyond the regex layer.
33. As an operator, I want PII (emails, phone numbers, card numbers, IPs) redacted from
    both the input and the final answer.
34. As an operator, I want a hardened system prompt that marks user content as untrusted
    data and forbids role changes and prompt disclosure.
35. As an operator, I want retrieved chunks wrapped in delimiters with a "this is data,
    not instructions" preamble (spotlighting), so that an indirect-injection payload
    hidden in a document can't hijack the answer.
36. As an operator, I want the model's output validated against the response schema, with
    an automatic LLM retry to fix malformed output, up to a limit.
37. As an operator, I want the security layers to run in a fixed, documented order.
38. As an operator, I want registration and login themselves rate-limited per IP.
39. As an operator, I want admin-only endpoints protected by an `is_admin` claim.

### Caching

40. As an operator, I want embeddings cached by content hash so that re-ingesting or
    re-querying the same text doesn't re-call the embedding API.
41. As an operator, I want intent classifications, generated SQL, SQL results, and full
    RAG answers each cached in their own tier with an appropriate TTL.
42. As an operator, I want the RAG-answer cache key to include the feature flags, so that
    toggling a technique doesn't return a stale answer.
43. As an SRE, I want a repeated identical query to come back near-instantly and flagged as
    a cache hit.
44. As an operator, I want a `/admin/cache/stats` endpoint reporting per-tier
    hit/miss/set counts and hit rate, and a way to clear the cache.
45. As an operator, I want the cache to fall back to an in-process store when Redis isn't
    configured, so that local development works without it.
46. As an operator, I want uploaded/ingested documents de-duplicated by source file name
    (checked against the vector store) so that re-ingesting a file is a no-op.

### Ingestion & data

47. As an operator, I want a seed command that runs the SQL migrations, creates demo
    users, and ingests the document corpus into the vector store.
48. As an operator, I want document parsing and chunking to handle PDF, DOCX, HTML, and
    text via a document-processing library.
49. As an operator, I want the noise corpus sampled by a fixed seed and a configurable
    size, with the signal corpus always ingested in full.
50. As an operator, I want the operational database seeded with a realistic synthetic
    Kubernetes schema (clusters, nodes, deployments, pods, incidents, alerts, on-call
    logs) and enough rows to make the demo queries meaningful.

### API surface & auth

51. As a new user, I want to register with a username and password and receive a JWT.
52. As a user, I want to log in and receive a JWT.
53. As an admin, I want to upload a PDF to be parsed, chunked, embedded, and indexed.
54. As an operator, I want a health endpoint that reports the status of every dependency
    (Postgres, vector store, Redis, LLM API, web-search API).

### Evaluation

55. As a developer, I want a fixed set of golden questions, each tagged with the technique
    it demonstrates and whether the naïve baseline is expected to fail it.
56. As a developer, I want to run the eval harness under a named flag profile (naïve,
    hybrid, hybrid+rerank, …, all) and get Ragas metrics (faithfulness, context
    precision, context recall, answer relevancy) plus source-overlap and
    forbidden-keyword checks.
57. As a developer, I want a diff between the naïve baseline run and the all-techniques run
    that shows the advanced pipeline scoring higher.
58. As a developer, I want eval results written to timestamped JSON files.

### Demo UI

59. As a user, I want a Streamlit app where I can register/log in, ask a question with all
    feature toggles exposed, and see the answer, sources, and metadata pane.
60. As a user, I want the Streamlit app to show the pending SQL and let me approve or
    reject it.
61. As a user, I want preset example questions covering every path and a dashboard that
    renders the latest eval results. **Split during issue #34 prototyping:** preset
    questions are covered by story 59 / issue #34; the eval-results dashboard is its own
    ticket, issue #54, blocked by #33 (its data source) — the two halves of this story
    don't depend on each other and were coupling unrelated tickets together.

### Packaging

62. As a developer, I want `docker compose up` to start Postgres, the vector store, and the
    app together, with the app waiting for its dependencies to be healthy.
63. As a developer, I want `make` targets for install, seed, run, test, lint, format, and
    each eval profile.
64. As a developer, I want lint (ruff), type-check (mypy), and unit tests to pass in CI.

## Implementation Decisions

### Stack

- **Language / runtime:** Python 3.12, managed with `uv`.
- **API:** FastAPI + uvicorn. JWT bearer auth (HS256, PyJWT), bcrypt password hashing.
- **Orchestration:** LangGraph with a Postgres checkpointer and `interrupt()` for the
  SQL-approval pause.
- **LLM:** OpenAI — `gpt-5.4-mini` for answer generation, `gpt-5.5` for grading/
  classification (deviates from an earlier gpt-4o/gpt-4o-mini split — ticket #23 owner
  decision: the grader should be at least as capable as the answer model, since it later
  judges CRAG/Self-RAG/eval quality — while keeping both models in a cost-conscious tier
  rather than the flagship gpt-5.6 generation).
- **Embeddings:** OpenAI `text-embedding-3-small` (1536-dim).
- **Vector store:** Qdrant, cosine distance.
- **Sparse retrieval:** in-process TF-IDF (scikit-learn) built by scrolling the Qdrant
  collection; fused with dense results via Reciprocal Rank Fusion (`k = 60`). **Owner
  decision (issue #34 scope):** re-enable `vector_store.hybrid_search`'s per-side
  candidate overfetch before fusion — currently dead-commented, so it fetches exactly
  `top_k` per side instead of a wider pool. Fixing this belongs to issue #34, not a new
  ticket, since it was found while prototyping that ticket's UI.
- **Reranker:** pluggable — local sentence-transformers `CrossEncoder`
  (`ms-marco-MiniLM-L-6-v2`) by default, or a hosted rerank API; failure falls back to the
  input order.
- **Relational DB:** PostgreSQL 16 — operational data, users, and LangGraph checkpoints.
- **Cache:** Upstash Redis (serverless), with an in-process TTL dict as fallback.
- **Web search:** Tavily (CRAG fallback only).
- **Security scanning:** `llm-guard` (PromptInjection, Toxicity, BanTopics, TokenLimit on
  input; Toxicity, BanTopics, Sensitive on output), with regex fallbacks for PII and
  injection when the library or a model is unavailable.
- **Document parsing:** a document-conversion library with a hybrid chunker
  (PDF/DOCX/HTML/TXT).
- **Logging:** loguru (structured JSON in production).
- **Tooling:** ruff, mypy, pytest; Ragas for eval.

### Modules and responsibilities

- **Config** — one settings object loaded from environment / `.env`, holding every tunable
  (models, URLs, cache TTLs, security thresholds, retrieval defaults).
- **Models** — all request/response schemas, including the L1 regex injection validators on
  free-text fields.
- **Auth & middleware** — password hashing, JWT mint/verify, `get_current_user` /
  `require_admin` dependencies, and a Redis sorted-set sliding-window rate limiter used for
  both per-IP (auth) and per-user (`/query`) limits.
- **Vector store service** — collection lifecycle, upsert, dense search; plus the sparse
  index build, sparse search, and hybrid (RRF) search.
- **Embedding service** — batched OpenAI embeddings with per-text cache read-through.
- **Document processor** — parse + chunk a file into `{text, source, page_number?}`.
- **Query cache service** — the multi-tier cache: tiers for embedding, intent, rag_answer,
  sql_gen, sql_result; SHA-256 keys; per-tier stats; Redis-or-memory.
- **LLM service** — thin wrappers for text generation and JSON-mode generation, returning
  text + token usage.
- **RAG service** — the retrieval-and-generate orchestrator: chooses the retrieval
  strategy from the flags (dense / sparse / hybrid / HyDE), applies reranking, runs the
  CRAG grade + fallback, builds the spotlighted context, generates, runs the Self-RAG
  loop, and reads/writes the rag_answer cache. Exposes a traced variant (no cache) for
  eval.
- **HyDE service** — generate N hypotheses + keep the original, embed all, search each,
  dedupe by normalised text keeping best score.
- **CRAG service** — LLM relevance grader → score; below threshold → web-search fallback;
  handles the empty-retrieval case. **Owner decision (issue #34 scope):** the web-fallback
  case needs to be distinguishable in `ChatResponse.metadata` so the demo UI can label it
  ("answered from the web, not our docs") instead of it looking like a normal corpus
  answer. `route` currently stays `"rag"` either way. Exact shape TBD alongside the rest
  of the UI design; fixing it belongs to issue #34.
- **Web search service** — Tavily query → chunks; explicit error when unconfigured.
- **Self-reflective service** — strict rubric critic → reflection score + regeneration
  decision + refined question; a `should_regenerate` gate bounded by a retry limit.
- **Router service** — LLM intent classifier (`sql` / `rag` / `hybrid`), cached.
- **SQL service** — schema introspection from `information_schema`, LLM SQL generation
  (cached), a `SELECT`-only guard with a keyword blocklist, execution with row
  serialisation and result caching.
- **Graph** — nodes: `route_intent`, `retrieve_rag`, `generate_sql_node`,
  `request_sql_approval` (interrupt), `execute_sql`, `generate_answer`, `finalize`;
  conditional edges on intent; Postgres checkpointer. Compiled once at startup.
- **Security modules** — input guard, content moderation + PII redaction, input
  restructuring, output validator (schema + LLM retry), token budget (per-user-per-day
  Redis counter), spotlighting, hardened system prompt.
- **Document dedup** — before ingesting a file, ask the vector store whether it already
  has chunks with that `source` name; if so, skip the file. No separate storage layer, no
  S3, no file hashing: the seed corpus is static and new files get unique names.
- **Eval** — golden schema + loader, flag profiles, a service invoker, the Ragas adapter,
  post-checks (forbidden keywords, source overlap), reporting/aggregation, and a CLI
  runner writing timestamped JSON.

### Demo UI (issue #34)

`spec.md`'s original stories 59–61 were written from a tutorial reference project, before
the UI's actual look and behaviour had been discussed. The following was decided by
prototyping several directions (throwaway HTML mockups, not final code — see the
prototype link on issue #34) and is what the real Streamlit implementation should build
from. Covers stories 59 and 60 only; story 61's eval-results panel has been split into its
own ticket (issue #54, blocked by #33) and is still unspecified there.

- **Layout — persistent sidebar, not a plain form or a pure chat window.** Compared three
  directions: (a) chat thread with toggles hidden in a settings drawer, (b) a form +
  result panel with no running transcript, (c) a persistent left rail (retrieval toggles +
  preset questions, always visible) next to a log-style transcript. **(c) won** — closest
  to idiomatic Streamlit (`st.sidebar` + `st.chat_message`), and keeps retrieval controls
  visible instead of hidden behind a settings icon. Transcript entries are a flat,
  rule-delineated log/timeline style, not chat bubbles.
- **Theme — warm neutral palette, dark by default.** Background/text/borders follow
  Claude.ai's own warm-grey palette (not a generic blue-grey or pure black/white), with a
  single terracotta accent (`#E58762` on dark, `#BD5B3A` on light) reserved for
  interactive/active elements only. Semantic colours (amber = pending approval, red =
  reject/critical, green = approved/healthy) are separate from the accent and never
  reused for anything else. Code/SQL blocks stay dark in both light and dark mode
  (a deliberate signature, like a real terminal). Two lighter alternatives (a warm cream
  and a muted "soothing" sage) were prototyped and are worth keeping as a
  theme-switcher option, not just thrown away — implementation should decide whether
  that's in scope for v1 or a follow-up.
- **Response rendering differs by `metadata.route`**, using a glyph instead of a text
  badge as the primary signal (shape encodes state, not just colour): `●` filled =
  single-source resolved answer (`rag`, `sql`), `◐` half-filled = `hybrid` (merged
  sources), `○` open = `sql_pending` (unresolved, awaiting approval). A pending-SQL entry
  shows the generated SQL in a code block, the explanation, and Approve/Reject actions —
  no answer text is rendered for that turn. A resolved answer shows the answer, a
  relevance-score meter, and source tags prefixed `[doc]` / `[sql]` so origin is visible
  at a glance without opening anything.
- **Debug/raw detail lives in a dedicated inspector, not inline.** Every resolved answer
  has an "Inspect response" action that opens a slide-over panel with a
  Formatted/Raw JSON toggle — formatted shows retrieval score, cache status, reflection
  info, and each retrieved chunk's source/score/text with a score meter; raw shows the
  literal `ChatResponse` JSON. Keeps the transcript itself scannable while still exposing
  everything `ResponseMetadata` carries, for users who want to audit an answer.
- **Flag-disabling logic** (the result of an explicit code audit — see issue #34 comments
  for the full flag × intent liveness table): only **one** conflict is knowable
  client-side before a question is even sent — `enable_hyde` always overrides
  `search_mode` (`hyde_search` only ever does a dense search internally, regardless of
  the mode picked) — so the UI greys out the search_mode control the instant HyDE is
  toggled on. Every other case (a question turning out to be pure-`sql` intent, so the
  whole retrieval rail was unused; `enable_adaptive_retrieval` skipping retrieval
  entirely for a general-knowledge question; `enable_rerank` silently no-op'ing below 2
  chunks) is **not** knowable until the response comes back, because intent
  classification is a server-side LLM call over the question text, uninfluenced by any
  flag. Rule: never pre-disable a control based on a guess — state honestly, on the
  response itself, what actually ran (via `route` and the inspector), rather than
  leaving the user to wonder whether a toggle they set did anything.
- **Preset example questions are kept**, scoped to exactly the three canonical examples
  already used in `HANDOFF.md` (one per path: RAG, SQL, hybrid). Purpose: this is
  explicitly a demo/test surface (issue #34 opens with "the visual tester"), the corpus
  is deliberately 95% noise, and a made-up question is likely to land on none of the
  three paths cleanly — presets guarantee a working example is always one click away.
- **Out of scope for this ticket:** story 61's eval-results panel — split into its own
  ticket, issue #54 ("Eval dashboard panel"), blocked by #33 (the eval harness that
  produces the JSON files it reads) — and Langfuse tracing (ticket #37, unrelated
  integration). Neither is designed yet; both need their own pass later.

### API contracts

- `POST /auth/register` → `{token}`; `POST /auth/login` → `{token}`. Both per-IP rate
  limited. `409` on duplicate, `401` on bad credentials.
- `POST /query` (bearer JWT) → a chat response: `answer`, `sources[]`, `retrieval_score`,
  optional `pending_sql {sql, query_id, explanation}`, `cache_hit`, and a `metadata` block
  (route, retrieved chunk previews, reflection telemetry). Body carries the six feature
  flags. When the graph interrupts for SQL approval, the response carries `pending_sql`
  and no answer.
- `POST /query/sql/execute` (bearer JWT) → resumes the graph for `query_id` with
  `approved: bool`; returns the same chat-response shape.
- `POST /documents/upload` (admin JWT) → parse, chunk, embed, index; deduped by source file name.
- `GET /admin/health` → per-dependency booleans + overall status.
- `GET /admin/cache/stats` (admin) → per-tier hits/misses/sets/hit_rate.
- `POST /admin/cache/clear` (admin) → clears caches.

### Security pipeline order (fixed)

L1 schema+regex → L4a JWT → L4b per-user rate limit → L6 token-budget check → L5 input
restructuring → L2 llm-guard input scan → L7a input moderation + PII redaction → graph
invoke (L3 hardened system prompt + L8 spotlighting inside generation) → L7b output
moderation + PII redaction → L9 output schema validation with LLM retry → consume token
budget → return.

### Operational SQL schema

Seven tables — `clusters`, `nodes`, `deployments`, `pods`, `incidents`, `alerts`,
`oncall_logs` — with foreign keys, seeded with a realistic volume of synthetic rows
(order of ~10k) generated with a fixed seed. The exact column set is taken from the seed
migration, not the prose description in the report (they differ). Users table is a
separate migration.

### Cache tiers and TTLs

embedding (7 d) · intent (24 h) · sql_gen (24 h) · sql_result (15 m) · rag_answer (1 h,
key includes flags). Keys are SHA-256 of the normalised input.

### Build order

The system is built as a chain of vertical slices, each adding one capability on top of a
working baseline, in roughly this order: project skeleton & config → request/response
models → auth + rate limiting → operational-DB schema & seeding → embeddings + vector
store + cache service → **native dense RAG end-to-end via `/query`** → hybrid search →
reranking → HyDE → CRAG + web fallback → Self-RAG → LangGraph state machine + intent
router → Text2SQL + human-in-the-loop approval → full multi-tier caching + document dedup →
the 9 security layers wired in fixed order → Ragas eval harness → Streamlit UI →
docker-compose end-to-end + docs. `/to-tickets` will turn this into the actual tickets
with blocking edges.

### Known clean-ups (decide per ticket)

- The eval goldens use short source filenames (`pods.html`) while ingestion records the
  full slugged name (`concepts__workloads__pods.html`) — reconcile in the eval slice.
- The intent router carries a hard-coded hint list from an unrelated corpus — drop it.
- `seed/docs/README.md` describes an unrelated e-commerce corpus — rewrite it for the
  Kubernetes corpus.
- Default `search_mode` is `dense` in the schema though some docs imply `hybrid` — keep
  `dense` as the schema default; the eval/UI profiles set it explicitly.

### Out of scope

- AWS deployment (CloudFormation / ECS / EFS / ALB / OIDC) — no infrastructure design or
  code behind it.
- The PDF's "optional add-ons" (multi-LLM, multi-modal, GraphRAG, agentic RAG,
  Langfuse, streaming SSE, multilingual).

## Testing Decisions

### Seams

- **Primary seam: the RAG service's traced entry point** (`run_rag_with_trace` / the
  no-cache variant). It returns the response plus the retrieved chunks, so retrieval
  quality is testable without HTTP or the graph. The eval harness invokes here.
- **Secondary seam: the FastAPI HTTP boundary** via `httpx` against the app — covers auth,
  the security pipeline order, the SQL-approval two-call flow, and admin endpoints.
- **Unit seams: each service module** — sparse index / RRF fusion, reranker ordering, HyDE
  dedupe, CRAG threshold behaviour, Self-RAG `should_regenerate` gate, SQL `SELECT`-only
  guard, cache key derivation and tier stats, token-budget arithmetic, PII regex fallback,
  L1 injection validators.

Prefer these existing seams; do not add new ones without agreeing them on the ticket.

### Approach

- `/tdd` at the unit seams above — write the behaviour test first (injection strings
  rejected, non-`SELECT` rejected, RRF ordering, cache hit on repeat, budget exhaustion →
  429).
- Integration tests at the HTTP seam for each end-to-end user story, using a test JWT and
  live local Postgres + Qdrant + (memory) cache.
- External APIs (OpenAI, Tavily, llm-guard models) are mocked in unit tests; integration
  and eval runs use the real APIs.
- The **eval harness is the regression gate for retrieval quality**: `make eval-baseline`
  vs `make eval-all` must show the all-techniques profile scoring higher on context recall
  and faithfulness; specific goldens tagged `expected_baseline: fail` must pass under their
  technique's profile.
- The **demo script** (five representative `curl` calls + a jailbreak) is the end-to-end
  acceptance for the whole system and must pass from a cold `docker compose up`.
- `ruff check`, `ruff format --check`, and `mypy` clean; full `pytest` green once at the
  end of each slice.
