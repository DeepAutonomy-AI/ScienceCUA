"""Persisted candidate contract for the single-source discovery workflow."""
from __future__ import annotations

import hashlib
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HtmlScopeComparison(BaseModel):
    """A model's comparison with another actually inspected entry."""

    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=4000)
    relation: Literal["broader", "narrower", "alternative"]
    reason: str = Field(min_length=1, max_length=1600)

    @field_validator("reason")
    @classmethod
    def nonempty_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A scope comparison requires a reason")
        return value


class HtmlSourceScope(BaseModel):
    """Model-owned scope judgment; the host binds its evidence to a capture."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["tutorial_collection", "user_manual", "documentation_collection", "single_page_manual"]
    scope_summary: str = Field(min_length=1, max_length=2000)
    entry_evidence_ids: list[str] = Field(min_length=1, max_length=16)
    included_sections: list[str] = Field(min_length=1, max_length=16)
    excluded_sections: list[str] = Field(default_factory=list, max_length=16)
    comparisons: list[HtmlScopeComparison] = Field(default_factory=list, max_length=4)
    comparison_note: str = Field(min_length=1, max_length=2400)
    limitations: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("scope_summary", "comparison_note")
    @classmethod
    def nonempty_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A scope decision requires a nonempty explanation")
        return value

    @field_validator("entry_evidence_ids", "included_sections", "excluded_sections", "limitations")
    @classmethod
    def bounded_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 600 for item in value):
            raise ValueError("Scope items must be nonempty and at most 600 characters")
        return value


class SourceCandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=4000)
    modality: Literal["pdf", "html"]
    source_type: Literal["official", "third_party", "unknown"] = "unknown"
    document_version: str = Field(default="unknown", max_length=200)
    version_relation: Literal["exact", "compatible", "older", "newer", "unknown"] = "unknown"
    version_evidence: str = Field(default="", max_length=6000)
    language: str = Field(default="unknown", max_length=100)
    platform: str = Field(default="unknown", max_length=200)
    root_assessment: str = Field(default="", max_length=6000)
    reason: str = Field(default="", max_length=6000)
    # Optional for historical candidates and PDFs; required by the host for new HTML saves.
    html_scope: HtmlSourceScope | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        # Transport/schema validation only. The Agent owns version and root judgments.
        value = value.strip()
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or any(char.isspace() for char in value)):
            raise ValueError("Candidate URL must be an HTTP(S) address without credentials")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("Candidate URL has an invalid port") from exc
        return value


SOURCE_DISCOVERY_TOOLS = ("save_source_candidate", "submit_source_candidates",
                          "list_discovered_sources", "inspect_source_candidate", "next_source_validation")


class NextSourceValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wait_seconds: float = Field(default=30, ge=0, le=45)


class DiscoveredSourcesPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12, ge=1, le=40)
    modality: Literal["all", "html", "pdf"] = "all"


class SourceCandidateInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    action: Literal["inspect", "skip"] = "inspect"
    reason: str = Field(default="", max_length=2000)

    _valid_url = field_validator("url")(SourceCandidateDraft.validate_url.__func__)

    @model_validator(mode="after")
    def require_skip_reason(self):
        if self.action == "skip" and not self.reason.strip():
            raise ValueError("Skipping a discovered source requires a concrete reason")
        return self


class SourceCandidateCheckpoint(BaseModel):
    """One complete Agent-verified candidate, saved before further exploration."""

    model_config = ConfigDict(extra="forbid")
    candidate: SourceCandidateDraft
    replaces_url: str | None = Field(default=None, max_length=4000,
        description="Explicitly replace a previously saved entry after inspecting a better scope; old checkpoints remain in audit.")

    @field_validator("replaces_url")
    @classmethod
    def validate_replaced_url(cls, value: str | None) -> str | None:
        return SourceCandidateDraft.validate_url(value) if value is not None else None


class SourceCandidate(SourceCandidateDraft):
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{16}$")
    # Host-only transport facts; models cannot supply these through Draft.
    verification: dict = Field(default_factory=dict)


def canonical_candidate_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


class SourceCandidateSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[SourceCandidateDraft] = Field(max_length=3)
    summary: str = Field(min_length=1, max_length=6000)
    search_notes: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("search_notes")
    @classmethod
    def bounded_notes(cls, value: list[str]) -> list[str]:
        if any(len(item) > 6000 for item in value):
            raise ValueError("Each search note must contain at most 6000 characters")
        return value

    @model_validator(mode="after")
    def unique_urls(self) -> "SourceCandidateSubmission":
        urls = [canonical_candidate_url(item.url) for item in self.candidates]
        if len(set(urls)) != len(urls):
            raise ValueError("Candidate URLs must be unique")
        return self

    def identified_candidates(self) -> tuple[SourceCandidate, ...]:
        return tuple(SourceCandidate(
            **item.model_dump(),
            candidate_id="candidate_" + hashlib.sha256(canonical_candidate_url(item.url).encode()).hexdigest()[:16],
        ) for item in self.candidates)
