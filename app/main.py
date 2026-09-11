"""FastAPI application factory.

`create_app()` builds the app; later tickets register their routers here
(`app.api.auth` in #20, `app.api.query` / `app.api.admin` in #23, ...). The
module-level `app` is what `uvicorn app.main:app` and `scripts/serve.py` serve.
"""

from fastapi import FastAPI

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise RAG", version="0.1.0")

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"service": "enterprise-rag", "status": "ok"}

    # Routers are added by their tickets:
    #   #20  app.api.auth
    #   #23  app.api.query, app.api.admin
    app.include_router(auth_router)
    app.include_router(query_router)
    app.include_router(admin_router)

    return app


app = create_app()
