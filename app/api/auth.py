"""`/auth/register` and `/auth/login` — username + password in, JWT out.

Both endpoints are rate limited **per IP** (register: `AUTH_REGISTER_RATE_LIMIT_PER_HOUR`,
login: `AUTH_LOGIN_RATE_LIMIT_PER_MIN`) so neither can be used to brute-force or to
flood the users table. Over budget → `429`. Duplicate username → `409`. Bad
credentials → `401`.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg2.errors
from fastapi import APIRouter, HTTPException, Request, status

from app import db
from app.config import settings
from app.middleware.auth import create_access_token, hash_password, verify_password
from app.middleware.rate_limiter import rate_limiter
from app.models import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """A real bcrypt digest verified against on a missing-user login so response
    time doesn't leak whether the username exists. Computed once, on first use."""
    return hash_password("not-a-real-password-placeholder")


def _client_ip(request: Request) -> str:
    # Trust only the real socket peer. The app is exposed directly (no reverse
    # proxy in docker-compose), so honouring a client-supplied X-Forwarded-For
    # here would let anyone spread their attempts across unlimited buckets and
    # walk straight past the per-IP limit.
    #
    # `request.client` is always set when served over TCP (uvicorn); the
    # "unknown" fallback only applies to non-TCP transports we don't deploy on.
    return request.client.host if request.client else "unknown"


def _enforce_ip_limit(request: Request, action: str, limit: int, window_seconds: int) -> None:
    if not rate_limiter.is_allowed_ip(_client_ip(request), action, limit, window_seconds):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many {action} attempts; try again later.",
            headers={"Retry-After": str(window_seconds)},
        )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, request: Request) -> TokenResponse:
    _enforce_ip_limit(
        request, "register", settings.auth_register_rate_limit_per_hour, window_seconds=3600
    )
    password_hash = hash_password(body.password)
    try:
        with db.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                "RETURNING id, username, is_admin",
                (body.username, password_hash),
            )
            row = cur.fetchone()
            if row is None:  # `INSERT ... RETURNING` without a conflict always yields a row
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Registration failed",
                )
            # Mint the token *before* the `with` block commits, so a failure here
            # rolls the insert back instead of leaving an account with no token
            # that can never be re-registered.
            token = create_access_token(row.id, row.username, row.is_admin)
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        ) from None
    return TokenResponse(token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    _enforce_ip_limit(request, "login", settings.auth_login_rate_limit_per_min, window_seconds=60)
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = %s",
            (body.username,),
        )
        row = cur.fetchone()

    if row is None:
        verify_password(body.password, _dummy_hash())  # equalise timing with the hit path
        raise _invalid_credentials()
    if not verify_password(body.password, row.password_hash):
        raise _invalid_credentials()

    token = create_access_token(row.id, row.username, row.is_admin)
    return TokenResponse(token=token)


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )
