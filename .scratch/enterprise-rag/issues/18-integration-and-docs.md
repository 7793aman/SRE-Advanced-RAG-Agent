# Integration: Docker end-to-end, notebooks, README & report

Type: task
Status: open
Blocked by: 16, 17

## Question

Final assembly — the whole stack runs from a cold `docker compose up`, and the repo's
own documentation matches what was built.

### Deliverables
- Finalize `Dockerfile` (torch CPU wheels, `psycopg[binary]`, copy `app/ scripts/ seed/`)
  and `docker-compose.yml` `app` service (env passthrough, volume mounts, `depends_on`
  healthchecks). `command: python scripts/serve.py`.
- `notebooks/` — copy the 5 teaching notebooks (`crag`, `hybrid_search`, `reranker`, `srag`,
  `text2sql`) from `_reference/notebooks/`.
- `README.md` — adapt `_reference/README.md` (quick start, endpoints, feature flags, demo
  script). Fix the stale bits noted on the map (`search_mode` default, `seed/docs/README.md`
  e-commerce text).
- `PROJECT_REPORT.md` — adapt `_reference/PROJECT_REPORT.md`; drop the AWS-deploy learning
  outcome and the screenshot placeholders (or fill them).
- `seed/docs/README.md` — rewrite for the actual K8s corpus (the reference's is stale
  e-commerce content).

### Reference
`_reference/{Dockerfile,docker-compose.yml,README.md,PROJECT_REPORT.md,notebooks/}`.

### Acceptance
- From a clean state: `docker compose up -d` → `docker compose exec app python scripts/seed_db.py`
  → the 5-call demo script in the README (RAG / SQL+approve / hybrid / CRAG-web / jailbreak)
  all return the expected results.
- `make lint` and `make test` pass.
- `git ls-files` in this repo covers every path in `git -C _reference ls-files` (minus
  `_reference/` itself and the noise corpus).
