# AGENTS.md — Enterprise RAG

An enterprise RAG system for Kubernetes IT-operations: a FastAPI service that answers
an SRE's natural-language questions via documentation retrieval, Text2SQL, or both,
behind a layered security pipeline and a multi-tier cache, orchestrated with LangGraph.

## How this project is built

- Work is tracked as tickets on the configured issue tracker (see `## Agent skills`).
- Each ticket is a vertical slice with acceptance criteria. When a ticket is picked up,
  the agent implements it (tests at the agreed seams), runs `/code-review`, and the owner
  reviews the diff to understand it.
- `spec.md` is the source of truth for intended behaviour. Tickets deliver its user stories.
- `_reference/` holds a working implementation of a similar system, kept locally as an
  implementation aid the agent may consult. It is gitignored and is not the spec.
- `projectReport.pdf` is the original design brief this system is based on.
- The noise corpus (802 MB) lives at `noisy_data 2/` in the repo root (gitignored).

## Agent skills

<!-- populated by /setup-matt-pocock-skills -->
