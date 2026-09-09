.PHONY: help install sync api serve test lint format typecheck check seed streamlit \
	eval eval-baseline eval-all eval-diff

help:
	@echo "Enterprise RAG — available commands"
	@echo ""
	@echo "  make install     — pin python 3.12, create venv, install all deps"
	@echo "  make sync        — re-sync deps with pyproject.toml"
	@echo "  make api         — run FastAPI with autoreload (:8000)"
	@echo "  make serve       — run FastAPI via scripts/serve.py (no reload)"
	@echo "  make test        — run pytest"
	@echo "  make lint        — ruff check"
	@echo "  make format      — ruff format"
	@echo "  make typecheck   — mypy app"
	@echo "  make check       — lint + typecheck + test"
	@echo ""
	@echo "  make seed        — migrations + demo users + doc ingestion   (ticket #21/#22)"
	@echo "  make streamlit   — demo UI (:8501)                            (ticket #34)"
	@echo "  make eval        — Ragas eval: baseline vs all + diff         (ticket #33)"

install:
	uv python pin 3.12
	uv venv --python 3.12
	uv sync --extra dev

sync:
	uv sync --extra dev

api:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

serve:
	uv run python scripts/serve.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app

check: lint typecheck test

# --- staged for later tickets -------------------------------------------------

seed:
	uv run python scripts/seed_db.py

streamlit:
	uv sync --extra ui && uv run streamlit run scripts/streamlit_app.py

eval-baseline:
	uv run python -m eval.run_ragas --profile naive

eval-all:
	uv run python -m eval.run_ragas --profile all

eval-diff:
	uv run python -m eval.diff

eval: eval-baseline eval-all eval-diff
