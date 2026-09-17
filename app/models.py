"""All request/response schemas for the API, plus the internal data shapes shared
between services.

`QueryRequest.question` carries the **L1** guardrail: a Pydantic + regex check that
rejects the crudest prompt-injection / script-injection payloads before any processing.
The reusable `validate_free_text` helper is exported so later free-text endpoints
(e.g. document upload) apply the same rule.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --- L1 guardrail: reject obvious injection payloads in free-text fields -----------

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(ignore\s+previous|ignore\s+above|forget\s+your\s+instructions)"),
    re.compile(r"(?i)(system\s*prompt|reveal\s+your\s+instructions|show\s+your\s+prompt)"),
    re.compile(r"(?i)(you\s+are\s+now|new\s+instructions|override\s+previous)"),
    re.compile(r"(?i)(<\s*script\b|javascript:)"),
    # HTML event-handler attributes (onload=, onerror=, ...). Named explicitly so
    # camelCase K8s terms like `terminationGracePeriodSeconds=30` don't false-positive.
    re.compile(
        r"(?i)\bon(?:load|error|click|focus|blur|submit|change|mouse\w+|key\w+|"
        r"abort|unload|toggle|input)\s*="
    ),
)
_NON_TEXT = re.compile(r"^[\W_]+$")


def validate_free_text(value: str, noun: str = "Input") -> str:
    """Strip, then reject empty, injection-shaped, or content-free text.

    Raises `ValueError` (surfaced by FastAPI as a 422) on any violation.
    """
    value = value.strip()
    if not value:
        raise ValueError(f"{noun} cannot be empty or whitespace only")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"{noun} contains potentially malicious content")
    if _NON_TEXT.match(value):
        raise ValueError(f"{noun} must contain actual text content")
    return value


# --- API: /auth ------------------------------------------------------------------


_BCRYPT_MAX_BYTES = 72  # bcrypt hashes only the first 72 bytes of the password


def _normalize_username(v: str) -> str:
    # Trim and lower-case so `Agent@demo.local` and `agent@demo.local` are the
    # same account. Both request models normalise identically, so a login always
    # matches what registration stored.
    return v.strip().lower()


def _check_password_bytes(v: str) -> str:
    # Cap on the *byte* length bcrypt actually sees, so a long multi-byte password
    # is rejected rather than silently truncated to a shared 72-byte prefix.
    if len(v.encode("utf-8")) > _BCRYPT_MAX_BYTES:
        raise ValueError(f"Password must be at most {_BCRYPT_MAX_BYTES} bytes")
    return v


class RegisterRequest(BaseModel):
    username: str
    password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def _clean_username(cls, v: str) -> str:
        v = _normalize_username(v)
        if not 3 <= len(v) <= 255:
            raise ValueError("Username must be 3–255 characters after trimming whitespace")
        return v

    @field_validator("password")
    @classmethod
    def _password_bytes(cls, v: str) -> str:
        return _check_password_bytes(v)


class LoginRequest(BaseModel):
    username: str
    password: str = Field(..., min_length=1)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, v: str) -> str:
        return _normalize_username(v)

    @field_validator("password")
    @classmethod
    def _password_bytes(cls, v: str) -> str:
        return _check_password_bytes(v)


class TokenResponse(BaseModel):
    token: str


# --- API: /query -----------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    search_mode: Literal["dense", "sparse", "hybrid"] = "dense"
    enable_rerank: bool = False
    enable_hyde: bool = False
    enable_crag: bool = True
    enable_self_reflective: bool = False
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _check_question(cls, v: str) -> str:
        return validate_free_text(v, "Question")


class RetrievedChunkPreview(BaseModel):
    text: str
    source: str
    score: float = 0.0


class ResponseMetadata(BaseModel):
    route: str = "rag"
    retrieved_chunks: list[RetrievedChunkPreview] = Field(default_factory=list)
    cache_hit: bool = False
    reflection_iterations: int = 0
    reflection_score: float | None = None
    refined_question: str | None = None


class PendingSQLBlock(BaseModel):
    sql: str
    query_id: str
    explanation: str = ""


class ChatResponse(BaseModel):
    answer: str = Field(..., min_length=0)
    sources: list[str] = Field(default_factory=list)
    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    pending_sql: PendingSQLBlock | None = None
    cache_hit: bool = False
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# --- Internal: shared between retrieval / grading / reflection services ----------


class RetrievedChunk(BaseModel):
    """Internal retrieval unit passed between services (vector store, HyDE, CRAG,
    reranker). Mirrors `RetrievedChunkPreview` today but is deliberately a separate
    type: the API-preview shape and the internal shape are free to diverge."""

    text: str
    source: str
    score: float = 0.0
    page_number: int | None = None


class CRAGEvaluation(BaseModel):
    relevance_score: float = 0.0
    relevance_label: str = ""
    confidence: float = 0.0
    reasoning: str = ""


class ReflectionResult(BaseModel):
    """Self-RAG reflection on a generated answer."""

    reflection_score: float = 0.0
    needs_regeneration: bool = False
    refined_question: str = ""
    reasoning: str = ""


class RetrievalDecision(BaseModel):
    """Self-RAG's adaptive-retrieval gate: whether a question needs a corpus
    search at all, or can be answered directly from general knowledge."""

    needs_retrieval: bool = True
    reasoning: str = ""
