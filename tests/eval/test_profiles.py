"""Unit seam: PROFILES is a plain dict, but its shape has to match
`QueryRequest`'s flag fields exactly, or `invoker.invoke()` silently sends flags
your app ignores. Pin that contract here rather than discovering it at runtime.
"""

from __future__ import annotations

from app.models import QueryRequest
from eval.profiles import PROFILES

_REQUIRED_KEYS = {
    "search_mode",
    "enable_rerank",
    "enable_hyde",
    "enable_crag",
    "enable_self_reflective",
    "enable_adaptive_retrieval",
    "top_k",
}


def test_every_profile_has_exactly_the_query_request_flag_keys() -> None:
    for name, flags in PROFILES.items():
        assert set(flags.keys()) == _REQUIRED_KEYS, f"profile {name!r} has the wrong flag keys"


def test_every_profile_is_a_valid_query_request() -> None:
    for flags in PROFILES.values():
        QueryRequest(question="placeholder", **flags)


def test_naive_profile_has_everything_off() -> None:
    naive = PROFILES["naive"]
    assert naive["search_mode"] == "dense"
    assert naive["enable_rerank"] is False
    assert naive["enable_hyde"] is False
    assert naive["enable_crag"] is False
    assert naive["enable_self_reflective"] is False
    assert naive["enable_adaptive_retrieval"] is False


def test_all_profile_has_everything_on() -> None:
    everything = PROFILES["all"]
    assert everything["search_mode"] == "hybrid"
    assert everything["enable_rerank"] is True
    assert everything["enable_hyde"] is True
    assert everything["enable_crag"] is True
    assert everything["enable_self_reflective"] is True
    assert everything["enable_adaptive_retrieval"] is True


def test_adaptive_retrieval_profile_isolates_only_that_flag() -> None:
    profile = PROFILES["adaptive_retrieval"]
    assert profile["enable_adaptive_retrieval"] is True
    assert profile["enable_rerank"] is False
    assert profile["enable_hyde"] is False
    assert profile["enable_crag"] is False
    assert profile["enable_self_reflective"] is False


def test_self_reflective_profile_isolates_only_that_flag() -> None:
    profile = PROFILES["self_reflective"]
    assert profile["enable_self_reflective"] is True
    assert profile["enable_rerank"] is False
    assert profile["enable_hyde"] is False
    assert profile["enable_crag"] is False
    assert profile["enable_adaptive_retrieval"] is False
