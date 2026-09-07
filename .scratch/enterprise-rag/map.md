# Enterprise RAG — build map

`wayfinder:map`

## Destination

A working, faithful, from-scratch reimplementation of the course project
`yashprogrammer/EnterpriseRAG_live` — the full FastAPI service: LangGraph state machine
routing RAG / Text2SQL / Hybrid, all 9 advanced-RAG / security modules, the 5-tier cache,
the Ragas eval harness, and the Streamlit demo — built by the owner one ticket at a time,
runnable end-to-end via `docker compose up` + `make seed` + the 5-call demo script.

"Done" = every file in `_reference/` (git tree, excluding `_reference/` itself) has a
hand-written equivalent in this repo, the demo script passes, and `make eval` shows the
advanced profiles beating the naïve baseline.

## Notes

**This is an execution map, not a planning map.** Wayfinder's "plan, don't do" default is
overridden: every ticket is a `task` that builds a slice of the system. Tickets are ordered
by a near-linear blocking chain that mirrors the reference's 10 feature commits.

**Working agreement:** spec → owner codes 100%. When a ticket is claimed, the agent first
expands the ticket's spec (reading the named reference files in detail), then the owner
writes the implementation, then the agent reviews it against `_reference/` and
`projectReport.pdf`. See `AGENTS.md`.

**Per-ticket skills:** none required by default. Use `mattpocock-skills:tdd` where a ticket
lists unit tests as acceptance criteria. Use `mattpocock-skills:code-review` for the review
step. Consult `CONTEXT.md` for vocabulary; don't drift to synonyms it says to avoid.

**Reference pointers:** each ticket names a commit hash in `_reference/` — check it out with
`git -C _reference show <hash>` / `git -C _reference checkout <hash> -- <path>` to see that
module in isolation. `git -C _reference log --oneline` lists them.

**Stack is fixed (faithful rebuild):** Python 3.12, FastAPI, LangGraph + Postgres
checkpointer, OpenAI GPT-4o / text-embedding-3-small, Qdrant, Postgres 16, Upstash Redis,
Tavily, llm-guard, Docling, scikit-learn TF-IDF for sparse, sentence-transformers
CrossEncoder for rerank. No provider swaps.

## Decisions so far

<!-- appended as tickets resolve -->

## Not yet specified

Almost nothing — the reference is fully visible, so all 19 build tickets are specified up
front. The only genuinely open items, to revisit once the modules they depend on exist:

- **Eval golden reconciliation** (surfaces in *Ragas eval harness*): `eval/seed_questions.yaml`
  references `golden_sources` like `pods.html`, but the real ingested filenames are
  `concepts__workloads__pods.html`. The fix (rename goldens, or map names, or re-slug the
  ingested `source`) is a call to make inside that ticket, not before.
- **Noise corpus wiring**: the owner has `noisy_data 2/` (820 files) at the repo root; the
  seed script wants `seed/docs/noisy_data/`. Symlink vs copy vs config override — decided in
  *Data: K8s ops schema & DB seeding*.
- **Reference cleanups to keep or drop**: the reference has a few leftovers from an earlier
  corpus (`router_service._DOCUMENT_HINTS` diffusion-LLM terms, stale `seed/docs/README.md`,
  empty `local_storage.py` with the class misplaced into `__init__.py`). Each ticket that
  touches one decides whether to faithfully copy or quietly fix.

## Out of scope

- **AWS deployment** (CloudFormation, ECS Fargate, EFS, ALB, GitHub Actions OIDC) — named in
  the report but no `infra/` directory exists in the reference git tree, so there is nothing
  to faithfully copy. Not part of this destination.
- **The report's "Optional Add-ons"** (multi-LLM, multi-modal, GraphRAG, agentic RAG,
  Langfuse, streaming SSE, etc.) — explicitly post-project.
