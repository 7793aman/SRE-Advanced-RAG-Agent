"""The app factory builds and serves a liveness route."""

from fastapi.testclient import TestClient

from app.main import app, create_app


def test_create_app_returns_fresh_instances() -> None:
    assert create_app() is not create_app()


def test_root_liveness() -> None:
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert resp.json() == {"service": "enterprise-rag", "status": "ok"}
