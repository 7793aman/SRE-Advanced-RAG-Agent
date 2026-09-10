"""Password hashing, JWT mint/verify, and the FastAPI auth dependencies.

* `hash_password` / `verify_password` — bcrypt via passlib.
* `create_access_token` / `decode_token` — HS256 JWTs (PyJWT). The token carries
  `sub` (user id, as a string), `username`, and `is_admin`, plus `iat`/`exp`.
* `get_current_user` — a dependency that turns the `Authorization: Bearer <jwt>`
  header into an `AuthenticatedUser`. Trust is claim-based: a well-signed,
  unexpired token is enough; we don't re-check the row on every request.
* `require_admin` — `get_current_user` plus an `is_admin` gate (403 otherwise).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from app.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer_scheme = HTTPBearer(auto_error=False)

_MIN_SECRET_LEN = 16


def _signing_secret() -> str:
    """The HS256 key, refusing to run with an unset or throwaway default.

    `config.py` defaults `jwt_secret` to `""` and docker-compose to `change-me`;
    either would let anyone forge an `is_admin` token, so fail loudly instead.
    """
    secret = settings.jwt_secret
    if len(secret) < _MIN_SECRET_LEN or secret.startswith("change-me"):
        raise RuntimeError(
            "JWT_SECRET is unset or too weak — set it to a long random string "
            f"(>= {_MIN_SECRET_LEN} chars)."
        )
    return secret


_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


class AuthenticatedUser(BaseModel):
    id: int
    username: str
    is_admin: bool = False


# --- passwords ------------------------------------------------------------------


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


# --- tokens --------------------------------------------------------------------


def create_access_token(
    user_id: int,
    username: str,
    is_admin: bool = False,
    expires_minutes: int | None = None,
) -> str:
    now = datetime.now(UTC)
    minutes = settings.jwt_expiration_minutes if expires_minutes is None else expires_minutes
    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": is_admin,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, _signing_secret(), algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _signing_secret(), algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise _CREDENTIALS_ERROR from exc


# --- dependencies -------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None:
        raise _CREDENTIALS_ERROR
    claims = decode_token(credentials.credentials)
    try:
        return AuthenticatedUser(
            id=int(claims["sub"]),
            username=claims["username"],
            is_admin=bool(claims.get("is_admin", False)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise _CREDENTIALS_ERROR from exc


def require_admin(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
