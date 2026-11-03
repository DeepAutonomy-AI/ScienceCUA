"""Script output DTOs. Canonical SourcePart/Evidence contracts stay unchanged."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import EvidenceNodeType


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnitRef(StrictOutput):
    resource_key: str
    unit_key: str
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def span(self):
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Supply both offsets or neither; offsets are Unicode [start,end).")
        if self.start_offset is not None and self.end_offset <= self.start_offset:
            raise ValueError("end_offset must exceed start_offset")
        return self


class ExtractedPart(StrictOutput):
    part_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    kind: str = "text"
    label: str = Field(min_length=1, max_length=300)
    parent_part_key: str | None = None
    ordinal: int = Field(ge=0)
    source_refs: list[UnitRef] = Field(default_factory=list)
    image_refs: list[UnitRef] = Field(default_factory=list)


class ExtractedNode(StrictOutput):
    node_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    title: str = Field(min_length=1, max_length=300)
    title_origin: Literal["source", "derived"] = "source"
    node_type: EvidenceNodeType = "section"
    parent_node_key: str | None = None
    part_keys: list[str] = Field(default_factory=list)


class Exclusion(StrictOutput):
    source_refs: list[UnitRef] = Field(min_length=1)
    reason_code: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=600)


class SourceGap(StrictOutput):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    resource_key: str | None = None
    url: str | None = None


class ExtractionOutput(StrictOutput):
    schema_version: Literal["adaptive-source-extraction.v1"] = "adaptive-source-extraction.v1"
    source_id: str
    parts: list[ExtractedPart] = Field(min_length=1)
    tree_nodes: list[ExtractedNode] = Field(min_length=1, max_length=512)
    exclusions: list[Exclusion] = Field(default_factory=list)
    gaps: list[SourceGap] = Field(default_factory=list)
    extraction_notes: str = Field(default="", max_length=4000)
