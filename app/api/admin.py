"""`/admin/health` (public) and `/admin/cache/*` (admin-only).

Story #54: health reports the status of every external dependency this
service actually needs to answer a question. Postgres, the vector store, and
the cache each get a real, live, cheap round-trip check; the LLM/web-search
"checks" only confirm a key is configured rather than spending money on a
live call every time someone hits this endpoint. The cache check
(`query_cache.ping()`) only ever fails when Redis is configured *and*
unreachable — story #45's in-process fallback always round-trips
successfully, so local dev without Redis still reports healthy.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import db
from app.config import settings
from app.middleware.auth import AuthenticatedUser, require_admin
from app.services.query_cache_service import query_cache
from app.services.vector_store import get_client

router = APIRouter(prefix="/admin", tags=["admin"])


def _check_postgres() -> bool:
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 — a health check reports down, never raises
        return False


def _check_vector_store() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:  # noqa: BLE001 — a health check reports down, never raises
        return False


@router.get("/health")
def health() -> dict[str, bool | str]:
    postgres_ok = _check_postgres()
    vector_store_ok = _check_vector_store()
    cache_ok = query_cache.ping()
    llm_api_ok = bool(settings.openai_api_key)
    web_search_api_ok = bool(settings.tavily_api_key)

    healthy = postgres_ok and vector_store_ok and cache_ok and llm_api_ok
    return {
        "postgres": postgres_ok,
        "vector_store": vector_store_ok,
        "cache": cache_ok,
        "llm_api": llm_api_ok,
        "web_search_api": web_search_api_ok,
        "status": "healthy" if healthy else "degraded",
    }


@router.get("/cache/stats")
def cache_stats(
    admin: AuthenticatedUser = Depends(require_admin),
) -> dict[str, dict[str, float | int]]:
    return query_cache.stats()


@router.post("/cache/clear")
def cache_clear(admin: AuthenticatedUser = Depends(require_admin)) -> dict[str, str]:
    query_cache.clear()
    return {"status": "cleared"}
