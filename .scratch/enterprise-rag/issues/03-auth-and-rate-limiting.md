# Auth, JWT & sliding-window rate limiting

Type: task
Status: open
Blocked by: 01, 02, 00

## Question

User registration/login with JWT, plus the Redis sliding-window rate limiter used by
both auth (per-IP) and `/query` (per-user).

### Deliverables
- `seed/migrations/001_create_users.sql` — `users(id, username UNIQUE, password_hash,
  is_admin DEFAULT false, created_at)`.
- `app/middleware/auth.py` — bcrypt `hash_password`/`verify_password`, `create_access_token`
  (HS256, `sub`/`exp`/`iat`/`is_admin`), `get_current_user` (HTTPBearer dependency),
  `require_admin`, `User` model.
- `app/middleware/rate_limiter.py` — `RateLimiter` class using an Upstash Redis sorted-set
  pipeline (`zremrangebyscore`/`zadd`/`zcard`/`expire`); `is_allowed_ip`, `is_allowed_user`.
- `app/api/auth.py` — `POST /auth/register` (per-IP hourly limit), `POST /auth/login`
  (per-IP per-minute limit), returns `{"token": ...}`. Wire router into `app/main.py`.

### Reference
commit `1d9e264` — `app/middleware/auth.py`, `app/middleware/rate_limiter.py`,
`app/api/auth.py`, `seed/migrations/001_create_users.sql`.

### Acceptance
- With Postgres + Redis up: register a user → get JWT; login → get JWT; wrong password → 401;
  duplicate register → 409; hammering `/auth/login` → 429.
