"""Unit seam: input restructuring (L5, story #31) — cap the question at a token ceiling."""

from __future__ import annotations

from app.security.input_restructuring import count_tokens, restructure_input


def test_text_under_the_ceiling_is_left_alone() -> None:
    assert restructure_input("Why is my pod in CrashLoopBackOff?", max_tokens=50) == (
        "Why is my pod in CrashLoopBackOff?"
    )


def test_text_over_the_ceiling_is_cut_to_exactly_the_ceiling() -> None:
    long_text = "Why is my pod in CrashLoopBackOff? " * 200

    result = restructure_input(long_text, max_tokens=10)

    assert count_tokens(result) == 10
    assert long_text.startswith(result)  # we keep the start and drop the tail
