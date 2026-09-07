# Project skeleton & config

Type: task
Status: open
Blocked by: —

## Question

Stand up the empty project shell so every later ticket has somewhere to write.

### Deliverables
- `pyproject.toml` — copy the reference's dependency set verbatim (it pins Phase 2–4 deps
  up front to avoid churn), plus `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`,
  hatchling build targeting `app`.
- `Makefile` — `install`, `sync`, `seed`, `api`, `streamlit`, `test`, `lint`, `format`,
  and the `eval-*` targets (targets can point at not-yet-existing modules).
- `Dockerfile`, `docker-compose.yml` (postgres:16 + qdrant/qdrant:v1.17.0 + app), `.env.example`.
- `.gitignore` (already exists — reconcile with reference's).
- `app/__init__.py`, `app/api/__init__.py`, `app/services/__init__.py`, `app/core/__init__.py`,
  `app/middleware/__init__.py`, `app/security/__init__.py`.
- `app/config.py` — the **full** `Settings(BaseSettings)` with every field the reference has
  (LLM, Qdrant, Postgres, cache TTLs, storage, Tavily, JWT, rate-limit, input-restructuring,
  security thresholds, retrieval defaults, Vanna, logging). Module-level `settings = Settings()`.
- `app/main.py` — FastAPI app factory. Routers can be commented out until they exist.
- `scripts/serve.py` — uvicorn entrypoint.

### Reference
`_reference/` commit `1d9e264` (baseline). Files: `pyproject.toml`, `Makefile`, `Dockerfile`,
`docker-compose.yml`, `.env.example`, `app/config.py`, `app/main.py`.

### Acceptance
- `make install` (or `uv sync --extra dev`) resolves.
- `python -c "from app.config import settings; print(settings.qdrant_url)"` works.
- `uvicorn app.main:app` boots with 0 routes (or a `/` stub) and no import errors.
