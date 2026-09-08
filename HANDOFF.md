# HANDOFF — Enterprise RAG

Everything a fresh agent (or person) needs to pick this project up cold. No prior
conversation assumed.

---

## 0. TL;DR

- **What:** build an "Enterprise RAG" service for Kubernetes IT-operations — a FastAPI app
  that answers an SRE's plain-English questions from documentation (RAG), an operational
  SQL database (Text2SQL), or both, orchestrated as a LangGraph state machine, behind a
  9-layer security pipeline and a 5-tier cache. Plus a Ragas eval harness and a Streamlit UI.
- **How:** it's a **learning rebuild**. There is a working reference implementation at
  `_reference/` (gitignored). The owner learns the codebase by having an agent implement
  one ticket at a time and then **explain it**.
- **State:** planning is done. `spec.md` is written. 18 build tickets exist as GitHub issues
  **#18–#35** on `github.com/7793aman/EnterpriseRAG`. **No application code exists yet.**
- **Next action:** implement issue **#19** (project skeleton). Start a fresh `claude` session
  in `/Users/aman/dev/RAG` and run `/implement 19` — or just "build issue #19".
- **Not this session's job:** cloud agents (the build needs local data + local Docker),
  wayfinder (no design fog — the reference defines everything).

---

## 1. What this project is (plain English)

A chatbot for engineers who keep Kubernetes running. An on-call engineer asks a question;
the system works out where the answer lives and fetches it:

- *"How do I debug a crashing pod?"* → **documentation** (the RAG path)
- *"Which cluster had the most P1 incidents last month?"* → **operational database** (the SQL path)
- *"Show P1 incidents on prod-us-east and the fix for each alert type"* → **both** (hybrid path)

The document corpus is **deliberately 95% noise** (820 unrelated technical PDFs) + 5% signal
(47 real Kubernetes docs). Naïve "find the most similar chunks" retrieval drowns in the
noise — so each advanced technique (hybrid search, reranking, HyDE, CRAG, Self-RAG) is
introduced as a fix for one specific way the noise breaks things. That failure-then-fix
arc is the whole pedagogical point.

**No model training.** "Creating embeddings" = calling the OpenAI embedding API to turn
text into vectors. Ingestion is a plain script (`scripts/seed_db.py`), run once. The
`notebooks/` are teaching scratchpads, not part of the running system.

### Architecture

```
  SRE → HTTP + JWT → FastAPI service
                       │
    ── SECURITY (in): validate · JWT · rate limit · token budget ·
       restructure · llm-guard scan · moderate + redact PII
                       │
                       ▼
               INTENT ROUTER (LLM)  → rag | sql | hybrid
                  │        │       │
         ┌────────┘        │       └────────┐
         ▼                 ▼                ▼
   RAG path          SQL path         both, merged
   retrieve (dense/  LLM writes SQL   ─────────────
   sparse/hybrid) →  → PAUSE for      run both legs
   rerank → HyDE →   human approval   and synthesise
   CRAG grade →      → run SELECT
   (web fallback) →  → rows
   generate → Self-RAG reflect/redo
         └────────────────┬────────────────┘
                          ▼
                  GENERATE ANSWER (LLM)
                          │
    ── SECURITY (out): moderate + redact PII · schema-validate (+ LLM retry)
                          ▼
                  answer + sources → SRE

  The whole in-box flow is a LangGraph state machine with a Postgres checkpointer,
  so the "pause for SQL approval" survives across two separate HTTP requests.

  Stores:  Qdrant (doc vectors) · Postgres (ops data + graph checkpoints) ·
           Upstash Redis (cache) · OpenAI (LLM + embeddings) · Tavily (web fallback)
```

---

## 2. Repository layout & where things live

| Path | What it is | In git? |
|---|---|---|
| `/Users/aman/dev/RAG` | the project repo | — |
| `github.com/7793aman/EnterpriseRAG` | private remote (`origin/main`) | — |
| `spec.md` | **source of truth** for intended behaviour | ✅ |
| `AGENTS.md` | working agreement + tracker config (auto-loaded every session) | ✅ |
| `CONTEXT.md` | domain glossary — use this vocabulary, don't drift | ✅ |
| `HANDOFF.md` | this file | ✅ |
| `docs/agents/` | issue-tracker / triage / domain config (from `/setup-matt-pocock-skills`) | ✅ |
| `seed/docs/true_data/` | 47 Kubernetes signal docs, pre-staged | ✅ |
| `seed/docs/noisy_data/` | empty (`.gitkeep`) — the noise corpus is wired in at ticket #21 | ✅ |
| `_reference/` | **working reference implementation** — read-only aid, consult freely | ❌ gitignored |
| `projectReport.pdf` | the original 16-page design brief this is based on | ❌ gitignored |
| `noisy_data 2/` | the 802 MB / 820-file noise corpus, at the repo root | ❌ gitignored |

There is **no `app/` yet** — that's what the tickets build.

---

## 3. The working agreement

From `AGENTS.md` (every session must follow this):

1. **One ticket per session.** Pick the lowest-numbered open issue whose "Blocked by"
   issues are all closed.
2. **The agent implements it** — code + tests at the seams `spec.md` names + `/code-review`
   + commit.
3. **Then the agent teaches it** — post a comment on the issue (what was built, key
   decisions, deviations), then in chat walk the owner through the design and answer
   questions.
4. **Only then close the issue.** Do not start the next ticket in the same session.

The owner is **learning the codebase**, not just shipping it. Explanation is part of the job.

---

## 4. Current state

**Done:**
- Repo initialised, pushed to GitHub (private).
- `/setup-matt-pocock-skills` run → GitHub issue tracker, default triage labels,
  single-context domain docs.
- `spec.md` written (problem, solution, ~64 user stories, implementation decisions,
  testing decisions/seams).
- `CONTEXT.md` glossary written.
- 18 build tickets created as GitHub issues **#18–#35** with blocking edges.
- 47 `true_data` signal docs staged in `seed/docs/true_data/`.
- Reference cloned to `_reference/`.

**Not done:**
- **No application code.** `app/`, `eval/`, `scripts/`, migrations — none exist yet.
- API keys not obtained, `.env` not created, Docker not brought up (that's ticket **#18**).
- 3 owner decisions still open (see §10).

**Quirk:** GitHub issues start at **#18**. A first attempt at issue creation hit a shell
bug (zsh array indexing) and produced 17 mis-paired issues; they were deleted, so #1–#17
are burned numbers. Cosmetic only.

---

## 5. The 18 build tickets (GitHub issues #18–#35)

Each issue body has: a short description, a **Scope** list, a **Done when** checklist, a
**Blocked by** line, and a **Spec:** pointer into `spec.md`.

| # | Title | Blocked by | Builds (reference files) |
|---|---|---|---|
| 18 | Setup: API keys & local infrastructure | — | `.env`, `docker compose up` (no code) |
| 19 | Project skeleton, config & models | — | `pyproject.toml`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `app/config.py`, `app/models.py`, `app/main.py`, `scripts/serve.py` |
| 20 | Auth, JWT & rate limiting | 18, 19 | `seed/migrations/001_create_users.sql`, `app/middleware/auth.py`, `app/middleware/rate_limiter.py`, `app/api/auth.py` |
| 21 | Operational DB schema & seeding | 19, 20 | `seed/migrations/003_seed_k8s_ops.sql`, `scripts/seed_db.py` |
| 22 | Embeddings, vector store & cache service | 18, 19 | `app/services/query_cache_service.py`, `embedding_service.py`, `vector_store.py` (dense), `document_processor.py` |
| 23 | Native RAG end-to-end (/query + /admin) | 20, 21, 22 | `app/services/llm_service.py`, `rag_service.py` (baseline), `app/api/query.py`, `app/api/admin.py`, `app/security/spotlighting.py`, `system_prompt.py` — **completes PDF Module 1** |
| 24 | Hybrid search (dense + sparse + RRF) | 23 | `app/services/sparse_vector_service.py`, `vector_store.py` (sparse/hybrid) — **Module 2** |
| 25 | Cross-encoder reranking | 24 | `app/services/reranking.py` — **Module 3** |
| 26 | HyDE | 25 | `app/services/hyde.py` — **Module 4** |
| 27 | CRAG + Tavily web fallback | 26, 18 | `app/services/crag.py`, `web_search.py` — **Module 5** |
| 28 | Self-RAG reflection loop | 27 | `app/services/self_reflective.py` — **Module 6** |
| 29 | LangGraph state machine + intent router | 28, 21 | `app/services/router_service.py`, `app/core/state.py`, `app/core/graph.py` — **Module 7a** |
| 30 | Text2SQL + human-in-the-loop approval | 29 | `app/services/sql_service.py`, graph SQL nodes, `/query/sql/execute` — **Module 7b** |
| 31 | Multi-tier caching + document dedup | 30 | `app/storage/*`, `app/services/doc_cache_service.py`, cache wiring — **Module 8** |
| 32 | 9-layer security pipeline | 31 | `app/security/input_guard.py`, `content_moderation.py`, `input_restructuring.py`, `output_validator.py`, `token_budget.py` + fixed-order wiring — **Module 9** |
| 33 | Ragas eval harness | 32 | `eval/*`, `eval/seed_questions.yaml` (40 K8s goldens) |
| 34 | Streamlit demo UI | 32 | `scripts/streamlit_app.py` |
| 35 | Docker end-to-end + docs + notebooks | 33, 34 | finalise `Dockerfile`/compose, `README.md`, `PROJECT_REPORT.md`, `notebooks/*` |

Frontier at handoff time: **#18 and #19** (nothing blocks them).

---

## 6. spec.md — the source of truth

`spec.md` is behaviour-level, not code-level. Sections:
- **Problem / Solution** — the "why"
- **User Stories** — ~64 numbered stories; each ticket's `Spec:` line says which it delivers
- **Implementation Decisions** — stack, module responsibilities, API contracts, the fixed
  security-layer order, the cache tiers/TTLs, the SQL schema note, build order, "known
  reference clean-ups", "out of scope"
- **Testing Decisions** — the **seams**:
  - primary: the RAG service's traced entry point (`run_rag_with_trace` / the no-cache
    variant) — retrieval quality testable without HTTP or the graph; the eval harness
    invokes here
  - secondary: the FastAPI HTTP boundary (via `httpx`)
  - unit: each service module
  Use `/tdd` at the unit seams. Do not invent new seams without agreeing on the ticket.

---

## 7. The reference implementation — how to use it

`_reference/` is a **complete, working** implementation of a very similar system. Treat it
as the answer key: when implementing a ticket, read the relevant files closely, then write
the equivalent here. It is **not** copied wholesale into commits — each ticket rebuilds its
slice, and the owner reviews it.

The reference is built as **10 clean feature commits** — one technique each. To see a
module in isolation:

```
git -C _reference log --oneline        # list the 10 commits
git -C _reference show <hash>          # see exactly what one module added
```

Commit → module map:
`1d9e264` native RAG · `2b68f19` hybrid · `57af24d` rerank · `1d4f8a0` HyDE ·
`bf3b02e` CRAG · `3a4cd05` Self-RAG · `d27a1d0` Text2SQL+LangGraph ·
`27171e5`(+`99dfa5c`,`b3761be`) caching · `3d7854a` security · `ee550c6` PROJECT_REPORT.

Note the reference has **no `tests/` directory** — zero tests upstream. This project adds
its own (see §10).

---

## 8. Stack & why each piece

| Layer | Tech | Why |
|---|---|---|
| Runtime | Python 3.12, `uv` | project standard |
| API | FastAPI + uvicorn | the web service / front door |
| Orchestration | LangGraph + Postgres checkpointer + `interrupt()` | branch / loop / **pause for SQL approval and resume across requests** |
| LLM | OpenAI GPT-4o (answers), GPT-4o-mini (grading/classification) | generation + the "judge" for CRAG/Self-RAG/eval |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | turn text into vectors for similarity search |
| Vector store | Qdrant (cosine) | holds every doc chunk as a vector; dense search |
| Sparse retrieval | **scikit-learn TF-IDF, in-process** (not BM25/FastEmbed) | exact-keyword matching; fused with dense via RRF (k=60) |
| Reranker | sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) local, or Voyage API | re-score top-20 candidates jointly with the query |
| Relational DB | PostgreSQL 16 | ops data (7 tables, ~10k synthetic rows) + LangGraph checkpoints + users |
| Cache | Upstash Redis (serverless), in-memory fallback | 5 tiers: embedding / intent / rag_answer / sql_gen / sql_result |
| Web search | Tavily | CRAG fallback when the corpus can't answer |
| Security | `llm-guard` (+ regex fallbacks) | prompt-injection / toxicity / ban-topics / PII |
| Doc parsing | Docling (hybrid chunker) | PDF / DOCX / HTML / TXT → chunks |
| Eval | Ragas | LLM-judged scores: faithfulness, context precision, context recall, answer relevancy |
| UI | Streamlit | visual tester |
| Tooling | ruff, mypy, pytest | lint / types / tests |

---

## 9. Known discrepancies: PDF vs reference code — **read before implementing**

The reference code is authoritative where it disagrees with `projectReport.pdf`:

1. **Sparse search is scikit-learn TF-IDF built in-process** by scrolling Qdrant — the PDF
   says "FastEmbed / BM25" and implies sparse vectors live in Qdrant. They don't.
2. **Reranker default is `ms-marco-MiniLM-L-6-v2`**, not "BGE" as the PDF says.
3. **`vanna` is a pinned dependency but never imported.** `sql_service.py` uses raw
   `psycopg2` + a plain LLM prompt, borrowing only `settings.vanna_model` as a model-name
   string. (See §10 — keep or drop.)
4. **The SQL schema in `003_seed_k8s_ops.sql` differs from the report's prose** — e.g.
   `environment` not `provider`, columns `alertname` / `rca_summary`, no `node_count`.
   Trust the migration file.
5. **`seed/docs/README.md` in the reference is stale** — it describes an e-commerce
   refund/warranty corpus. The real corpus is Kubernetes docs. Rewrite it (ticket #35).
6. **`eval/seed_questions.yaml` golden source names are short** (`pods.html`) but ingested
   `source` values are the full slug (`concepts__workloads__pods.html`). Reconcile in
   ticket #33 (rename goldens / substring-match / re-slug — pick one).
7. **`router_service.py` has a leftover `_DOCUMENT_HINTS` list** of diffusion-LLM terms
   from an unrelated corpus. Drop it (ticket #29).
8. **`app/storage/__init__.py` contains the `LocalStorage` class**; `local_storage.py` is
   empty. Put the class where it belongs (ticket #31).
9. **`app/models.py` `search_mode` defaults to `"dense"`** though some README examples say
   `hybrid`. Keep `dense` as the schema default; eval/UI profiles set it explicitly.
10. **`/documents/upload` is documented in the README/PDF but not implemented** in the
    reference. See §10 and §11.
11. **`scripts/data_pipeline/` is referenced** (`make seed-data`, some pyproject dev deps)
    **but does not exist** in the reference. The `true_data` docs and `003_*.sql` are
    pre-generated committed artifacts. `make seed-data` will be a dead target — drop it or
    make it a no-op (ticket #35).
12. **No AWS deploy exists.** The PDF lists ECS/EFS/ALB/OIDC as a learning outcome and the
    README mentions `infra/cloudformation.yaml` + `docs/DEPLOYMENT_GUIDE.md` — none of it is
    in the reference. Marked out of scope in `spec.md`.

---

## 10. Open decisions the owner must make

Ask the owner before the tickets they affect:

| Decision | Affects | Recommendation |
|---|---|---|
| **Build `/documents/upload`?** The endpoint is documented but not in the reference. spec.md has a user story for it. | #22 (or a micro-ticket) | **Build it** — ~30 lines on top of `document_processor` + `embedding_service` + `vector_store` + `doc_cache_service`. Completes the story. |
| **Keep the added test suite?** The reference has zero tests; spec.md + every ticket's acceptance criteria require unit tests. | every ticket, ~+20–30% time | **Keep** — the owner is learning; tests are how each slice gets verified. |
| **Keep `vanna` as a dependency?** Pinned but unused. | #19 (installs a heavy dep) | Either is fine — keep for pyproject fidelity, or drop it and the 3 `VANNA_*` settings. |
| **Skip AWS deployment?** A stated PDF outcome with nothing upstream to copy. | scope | **Skip** — no reference material exists. Revisit as a separate effort if wanted. |

---

## 11. Verification findings (coverage check done at handoff)

Every PDF module and every one of the 42 `app/` files + eval + scripts + seed maps to a
ticket. Gaps found:

- **Real gap:** `/documents/upload` — not in reference, not in a ticket (decision above).
- **Unavoidable gaps** (nothing upstream to copy): AWS deploy, `scripts/data_pipeline/`,
  `docs/DEPLOYMENT_GUIDE.md`.
- **Scope addition:** the test suite (reference has none).
- **Tidy:** add a direct `#18` blocking edge to `#21` (transitively covered via #20, but
  GitHub won't show it). Note the dead `make seed-data` target in #35.

---

## 12. How to run / build / test

Nothing runs yet. Once ticket #19 lands, the intended commands (from the reference Makefile):

```
make install        # uv venv + uv sync --extra dev
docker compose up -d postgres qdrant
cp .env.example .env && $EDITOR .env      # add real keys (ticket #18)
make seed           # migrations + demo users + ingest docs into Qdrant
make api            # uvicorn app.main:app on :8000
make streamlit      # Streamlit UI on :8501
make test           # pytest
make lint / make format
make eval-baseline / make eval-all / make eval-diff
```

Demo users after `make seed`: `agent@demo.local` / `agent123`, `admin@demo.local` / `admin123`.

---

## 13. Session hygiene

- **Always start the session in `/Users/aman/dev/RAG`** (`cd` there, then `claude`). Slash
  commands (`/implement`, `/code-review`) act on the launch directory. A session started
  elsewhere will target the wrong repo.
- **One fresh session per ticket.** Do not carry a single conversation across many tickets
  and lean on compaction — the durable memory is: the GitHub issues (+ their closing
  comments), `spec.md`, `CONTEXT.md`, `HANDOFF.md`, and git history.
- **Do not use cloud agents** for the build — they can't see `noisy_data 2/`, `_reference/`,
  or local Docker. This is local work.
- **Do not** run `/wayfinder` — there is no design fog; the reference defines everything.
  spec.md + the issues already are the plan.

---

## 14. How we got here (brief)

The owner bought an agentic-engineering course; "Enterprise RAG" is one project, shipped
fully implemented. Rather than watch the videos, the plan is to rebuild it ticket-by-ticket
with an agent implementing and explaining.

Path taken: `/wayfinder` was run first (wrong tool — it's for problems with unknowns; this
has none). It produced a local-file map + 19 draft tickets, since no tracker was configured
yet. Those were **deleted** and replaced with a hand-written `spec.md` (richer than
`/to-spec` would produce fresh, because the whole reference was read in that session).
Then `/setup-matt-pocock-skills` set GitHub as the tracker, and the 18 issues were created
directly. So: no `wayfinder:map` issue exists (by design), and issue numbers start at #18.

The standard flow from here is the matt-pocock **`/implement`** per ticket.

---

## 15. Concept primer (for explaining to the owner)

- **RAG** = Retrieval-Augmented Generation: retrieve relevant text → put it in the prompt →
  LLM generates a grounded answer. The "documentation path" here.
- **Agentic** (here) = an LLM-driven workflow with decision points (the router), tool use
  (web search, SQL), self-correcting loops (CRAG, Self-RAG), and a human checkpoint (SQL
  approval) — wired as a graph. Not an autonomous free-roaming loop.
- **"Hybrid"** is used for two unrelated things: the **hybrid path** (needs docs + database)
  and **hybrid search** (dense + sparse retrieval fused with RRF, inside the docs path).
- **Dense vs sparse search:** dense matches meaning ("progressive rollout" ≈ "rolling
  update"); sparse matches exact words (`CrashLoopBackOff`, `kubectl`).
- **Embeddings ≠ training.** An embedding is an API call that returns a vector. No model is
  trained anywhere in this project.
- **Noisy data:** 95% irrelevant docs are loaded on purpose so naïve retrieval fails and
  each technique has to demonstrably rescue the signal.
- **Eval / Ragas:** a fixed ~40-question answer key (`golden_sources` + `golden_answer_keywords`
  a developer wrote once). The pipeline "takes the exam"; an LLM judge scores each answer
  0–1 on faithfulness / context precision / context recall / answer relevancy; you compare
  the naïve profile vs the all-techniques profile to prove the techniques help. Runs
  offline via `make eval`, never on real traffic.
- **The SQL data is synthetic** — a script generates ~10k rows across 7 tables (clusters,
  nodes, deployments, pods, incidents, alerts, oncall_logs). No real Kubernetes, no AWS.
  It stands in for the incident tracker / CMDB / paging system that in reality *are*
  SQL-backed.
