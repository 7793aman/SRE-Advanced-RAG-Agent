"""Unit seam: password hashing + JWT mint/verify + the auth dependencies."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.config import settings
from app.middleware.auth import (
    AuthenticatedUser,
    create_access_token,
    decode_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)


def test_hash_is_not_plaintext_and_verifies() -> None:
    digest = hash_password("correct horse battery staple")
    assert digest != "correct horse battery staple"
    assert verify_password("correct horse battery staple", digest) is True
    assert verify_password("wrong password", digest) is False


def test_hash_is_salted() -> None:
    assert hash_password("same") != hash_password("same")


def test_token_round_trips_claims() -> None:
    token = create_access_token(user_id=42, username="agent@demo.local", is_admin=True)
    claims = decode_token(token)
    assert claims["sub"] == "42"
    assert claims["username"] == "agent@demo.local"
    assert claims["is_admin"] is True


def test_expired_token_is_rejected() -> None:
    past = datetime.now(UTC) - timedelta(minutes=1)
    token = jwt.encode(
        {"sub": "1", "username": "u", "is_admin": False, "exp": past},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_minting_refuses_a_weak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "jwt_secret", "change-me")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        create_access_token(user_id=1, username="u")


def test_token_signed_with_other_secret_is_rejected() -> None:
    token = jwt.encode({"sub": "1"}, "a-completely-different-secret-key-32b+", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_get_current_user_from_valid_token() -> None:
    token = create_access_token(user_id=7, username="u", is_admin=False)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = get_current_user(creds)
    assert user == AuthenticatedUser(id=7, username="u", is_admin=False)


def test_get_current_user_without_credentials_is_401() -> None:
    with pytest.raises(HTTPException) as exc:
        get_current_user(None)
    assert exc.value.status_code == 401


def test_require_admin_allows_admin_and_blocks_non_admin() -> None:
    admin = AuthenticatedUser(id=1, username="a", is_admin=True)
    assert require_admin(admin) is admin

    with pytest.raises(HTTPException) as exc:
        require_admin(AuthenticatedUser(id=2, username="b", is_admin=False))
    assert exc.value.status_code == 403
