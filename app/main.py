"""FastAPI application factory.

`create_app()` builds the app; later tickets register their routers here
(`app.api.auth` in #20, `app.api.query` / `app.api.admin` in #23, ...). The
module-level `app` is what `uvicorn app.main:app` and `scripts/serve.py` serve.
"""

from fastapi import FastAPI

from app.api.auth import router as auth_router


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise RAG", version="0.1.0")

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"service": "enterprise-rag", "status": "ok"}

    # Routers are added by their tickets:
    #   #20  app.api.auth
    #   #23  app.api.query, app.api.admin
    app.include_router(auth_router)

    return app


app = create_app()
