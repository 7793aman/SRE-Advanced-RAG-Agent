# Prereqs: API keys & local infrastructure

Type: task
Status: open
Blocked by: —

## Question

Get every external dependency the build needs into place, and record where the
credentials live. This unblocks every ticket that calls a paid service or a container.

### Checklist (owner does this)
- **OpenAI API key** — has GPT-4o + GPT-4o-mini + text-embedding-3-small access. Put in `.env`
  as `OPENAI_API_KEY`.
- **Upstash Redis** — create a free serverless Redis DB; copy REST URL + token to
  `UPSTASH_REDIS_URL` / `UPSTASH_REDIS_TOKEN`.
- **Tavily API key** — free tier; `TAVILY_API_KEY`. (Only strictly needed at ticket 10, but
  get it now.)
- **`JWT_SECRET`** — any long random string.
- **Docker Desktop running.** `docker compose up -d postgres qdrant` from the repo root once
  ticket 01 has written `docker-compose.yml` (so this ticket may finish in two sittings, or
  the compose file can be hand-created first).
- Confirm: `psql postgresql://postgres:postgres@localhost:5432/adv_rag -c '\l'` connects;
  `curl localhost:6333/healthz` returns ok.

### Answer (record on resolution)
- Which `.env` path holds the keys (should be `/Users/aman/dev/RAG/.env`, gitignored).
- Postgres + Qdrant container names / ports confirmed up.
- Any key that's on a low quota worth knowing about.
