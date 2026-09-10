"""HTTP seam: `/auth/register` and `/auth/login` end to end against live Postgres."""

from fastapi.testclient import TestClient

from app.middleware.auth import decode_token


def _register(
    client: TestClient, username: str = "sre@demo.local", password: str = "hunter2!!"
) -> object:
    return client.post("/auth/register", json={"username": username, "password": password})


def test_register_returns_a_usable_jwt(client: TestClient, clean_users: None) -> None:
    resp = _register(client)
    assert resp.status_code == 201
    token = resp.json()["token"]
    claims = decode_token(token)
    assert claims["username"] == "sre@demo.local"
    assert claims["is_admin"] is False


def test_duplicate_registration_is_409(client: TestClient, clean_users: None) -> None:
    assert _register(client).status_code == 201
    resp = _register(client)
    assert resp.status_code == 409


def test_registration_and_login_are_case_insensitive(client: TestClient, clean_users: None) -> None:
    assert _register(client, username="SRE@Demo.Local", password="hunter2!!").status_code == 201
    # same identity, different case → duplicate
    assert _register(client, username="sre@demo.local", password="hunter2!!").status_code == 409
    resp = client.post("/auth/login", json={"username": "Sre@Demo.Local", "password": "hunter2!!"})
    assert resp.status_code == 200


def test_login_with_correct_password_returns_jwt(client: TestClient, clean_users: None) -> None:
    _register(client, password="hunter2!!")
    resp = client.post("/auth/login", json={"username": "sre@demo.local", "password": "hunter2!!"})
    assert resp.status_code == 200
    assert decode_token(resp.json()["token"])["username"] == "sre@demo.local"


def test_login_with_wrong_password_is_401(client: TestClient, clean_users: None) -> None:
    _register(client, password="hunter2!!")
    resp = client.post("/auth/login", json={"username": "sre@demo.local", "password": "not-it"})
    assert resp.status_code == 401


def test_login_for_unknown_user_is_401(client: TestClient, clean_users: None) -> None:
    resp = client.post("/auth/login", json={"username": "ghost", "password": "whatever1"})
    assert resp.status_code == 401


def test_hammering_login_trips_the_per_ip_limit(client: TestClient, clean_users: None) -> None:
    _register(client, password="hunter2!!")
    body = {"username": "sre@demo.local", "password": "wrong"}
    # AUTH_LOGIN_RATE_LIMIT_PER_MIN defaults to 5
    codes = [client.post("/auth/login", json=body).status_code for _ in range(7)]
    assert codes[:5] == [401] * 5
    assert codes[5] == 429
    assert codes[6] == 429


def test_hammering_register_trips_the_per_ip_limit(client: TestClient, clean_users: None) -> None:
    # AUTH_REGISTER_RATE_LIMIT_PER_HOUR defaults to 3
    codes = [
        _register(client, username=f"u{i}@demo.local", password="hunter2!!").status_code
        for i in range(5)
    ]
    assert codes[:3] == [201, 201, 201]
    assert codes[3] == 429
    assert codes[4] == 429


def test_short_password_is_rejected_by_schema(client: TestClient, clean_users: None) -> None:
    resp = client.post("/auth/register", json={"username": "x@demo.local", "password": "short"})
    assert resp.status_code == 422
