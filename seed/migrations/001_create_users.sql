-- 001_create_users.sql — the users table for auth (#20).
--
-- One row per registered user. `username` is the login identifier (the demo
-- seed uses email-shaped values like `agent@demo.local`); `password_hash` is a
-- bcrypt digest, never the plaintext. `is_admin` gates the /admin endpoints and
-- rides in the JWT as the `is_admin` claim.

CREATE TABLE IF NOT EXISTS users (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username      TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    is_admin      BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
