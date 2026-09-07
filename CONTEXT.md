# CONTEXT — Enterprise RAG

A Kubernetes IT-Operations copilot: one FastAPI service that answers an SRE's
natural-language questions by routing them to documentation retrieval (RAG),
operational-data queries (Text2SQL), or both (Hybrid) — behind a 9-layer security
stack and a 5-tier Redis cache, orchestrated as a LangGraph state machine.

## Glossary

**Signal / true_data**
The ~47 real Kubernetes documentation files (`seed/docs/true_data/`) the system is
*supposed* to retrieve. The 5% of the corpus that matters.
_Avoid_: "the docs" (ambiguous with noise), "knowledge base" (that's signal + noise).

**Noise / noisy_data**
~820 unrelated technical PDFs (`noisy_data 2/` in the repo root). Deliberately 95% of
the corpus so naïve top-k retrieval drowns — every advanced technique has to earn its
place by rescuing signal from noise.
_Avoid_: "distractors" (use "noise"), "junk".

**Intent**
The router's classification of a question: `rag`, `sql`, or `hybrid`. Decides which
path the LangGraph state machine takes.
_Avoid_: "route" (that's the effect), "mode" (that's `search_mode`).

**search_mode**
How retrieval runs within the RAG path: `dense`, `sparse`, or `hybrid` (dense + sparse
fused with Reciprocal Rank Fusion). A per-request flag, independent of intent.
_Avoid_: "retrieval mode" is fine; don't call it "intent".

**Flags**
The per-request `QueryRequest` toggles — `search_mode`, `enable_rerank`, `enable_hyde`,
`enable_crag`, `enable_self_reflective`, `top_k` — that turn advanced techniques on and
off. Also the eval harness's "profiles".
_Avoid_: "options", "settings" (that's `app/config.py`).

**Module**
One self-contained advanced technique, matching one commit in the reference repo:
Native RAG, Hybrid Search, Reranking, HyDE, CRAG, Self-RAG, Text2SQL, Caching, Security.
The build adds one module at a time on top of a working baseline.
_Avoid_: "phase", "lesson" (the reference uses "commit" / "module").

**The 9 layers (L1–L9)**
The fixed-order defensive pipeline every `/query` request traverses: L1 Pydantic+regex,
L4a JWT, L4b rate limit, L6 token budget, L5 input restructuring, L2 llm-guard scan,
L7a input moderation+PII, L3 hardened system prompt, L8 spotlighting, L7b output
moderation+PII, L9 output schema validation. Numbered by the course's threat model,
not by execution order.
_Avoid_: renumbering them; the out-of-order numbering is intentional and matches the report.

**Reference**
The cloned upstream repo at `_reference/` (gitignored, read-only). The source of truth
for "what does the finished code do". Not to be copied wholesale — each ticket rebuilds
its slice.
