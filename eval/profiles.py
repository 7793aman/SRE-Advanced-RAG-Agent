"""Named flag profiles (issue #33, story 56). Each key here must match a field on
`app.models.QueryRequest` exactly — `tests/eval/test_profiles.py` pins that.
"""

from __future__ import annotations

PROFILES: dict[str, dict] = {
    "naive": {
        "search_mode": "dense",
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "sparse_only": {
        "search_mode": "sparse",
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "hybrid": {
        "search_mode": "hybrid",
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "hybrid+rerank": {
        "search_mode": "hybrid",
        "enable_rerank": True,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "hybrid+rerank+hyde": {
        "search_mode": "hybrid",
        "enable_rerank": True,
        "enable_hyde": True,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "hybrid+rerank+crag": {
        "search_mode": "hybrid",
        "enable_rerank": True,
        "enable_hyde": False,
        "enable_crag": True,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    # Isolates adaptive retrieval the same way sparse_only/hybrid isolate their
    # own technique — everything else off, so a pass/fail here is unambiguous.
    "adaptive_retrieval": {
        "search_mode": "dense",
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": False,
        "enable_adaptive_retrieval": True,
        "top_k": 5,
    },
    # Isolates the self-reflective critique/regenerate loop the same way — was
    # missing before, which meant self_rag goldens could only ever be checked
    # against "all" (where an unrelated adaptive_retrieval interaction can mask
    # whether self-reflection itself is working).
    "self_reflective": {
        "search_mode": "hybrid",
        "enable_rerank": False,
        "enable_hyde": False,
        "enable_crag": False,
        "enable_self_reflective": True,
        "enable_adaptive_retrieval": False,
        "top_k": 5,
    },
    "all": {
        "search_mode": "hybrid",
        "enable_rerank": True,
        "enable_hyde": True,
        "enable_crag": True,
        "enable_self_reflective": True,
        "enable_adaptive_retrieval": True,
        "top_k": 5,
    },
}
