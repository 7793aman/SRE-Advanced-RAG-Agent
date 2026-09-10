"""Schema-level rules for the auth request bodies."""

import pytest
from pydantic import ValidationError

from app.models import LoginRequest, RegisterRequest


def test_username_is_trimmed_then_length_checked() -> None:
    assert RegisterRequest(username="  sre@demo.local  ", password="hunter2!!").username == (
        "sre@demo.local"
    )
    # "  ab  " passes a naive char count but is only 2 chars once trimmed
    with pytest.raises(ValidationError):
        RegisterRequest(username="  ab  ", password="hunter2!!")


def test_login_username_is_trimmed_to_match_registration() -> None:
    assert LoginRequest(username="  sre@demo.local  ", password="x").username == "sre@demo.local"


def test_username_is_lowercased_on_both_models() -> None:
    assert RegisterRequest(username="SRE@Demo.Local", password="hunter2!!").username == (
        "sre@demo.local"
    )
    assert LoginRequest(username="SRE@Demo.Local", password="x").username == "sre@demo.local"


def test_password_cap_is_measured_in_bytes_not_chars() -> None:
    # 40 three-byte characters = 120 bytes > 72, even though it's < 72 "characters"
    with pytest.raises(ValidationError):
        RegisterRequest(username="sre@demo.local", password="€" * 40)


def test_short_password_rejected() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(username="sre@demo.local", password="short")
