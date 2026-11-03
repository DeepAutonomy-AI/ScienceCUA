from __future__ import annotations

import hashlib

import pytest

from app.knowledge import KnowledgeWorkspace
from app.models import ReferenceCandidate, ReferencePresentation


def reference_with_presentation(content: str | None = None) -> ReferenceCandidate:
    reference = ReferenceCandidate(
        reference_id="R1", corpus_snapshot_id="corpus_presentation_test",
        title="Documented operators", description="Check operator syntax and precedence.",
        reference_mode="source_catalog", source_catalog_id="C1",
        entries=[{
            "entry_id": "RE1", "title": "Source operator row", "content": "** ^ 2 power",
            "source_evidence_id": "E1",
            "citations": [{"evidence_id": "E1", "quote": "** ^ 2 power"}],
        }],
        citations=[{"evidence_id": "E1", "quote": "** ^ 2 power"}],
    )
    if content is not None:
        reference.presentation = ReferencePresentation(
            content=content, source_artifact_id="art_" + "a" * 24,
            source_sha256="b" * 64,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            evidence_ids=["E1"],
        )
    return reference


def workspace_for(reference: ReferenceCandidate, stage: str = "task_trajectory") -> KnowledgeWorkspace:
    workspace = KnowledgeWorkspace.__new__(KnowledgeWorkspace)
    workspace.stage = stage
    workspace.references_by_id = {reference.reference_id: reference}
    workspace.submitted_references = []
    workspace.evidence_by_id = {}
    workspace.trajectory_packet_full_evidence_ids = set()
    return workspace


@pytest.mark.parametrize("stage", ["task_trajectory", "trajectory_inference"])
def test_formatted_lookup_returns_presentation_without_changing_source_entries(stage):
    text = "## Operators\n\n| Operator | Alias | Precedence | Meaning |\n| --- | --- | --- | --- |\n| ** | ^ | 2 | power |\n"
    reference = reference_with_presentation(text)
    before = reference.model_dump_json()
    result = workspace_for(reference, stage).read_reference("R1")
    assert result["formatting_mode"] == "agent_formatted"
    assert result["content"] == text
    assert result["citation_ids"] == ["E1"]
    assert result["next_offset"] is None
    assert not result["content_truncated"]
    assert reference.model_dump_json() == before


def test_formatted_lookup_paginates_in_lines_without_losing_or_duplicating_content():
    text = "".join(f"Documented row {index}\n" for index in range(45))
    workspace = workspace_for(reference_with_presentation(text))
    parts = []
    offset = 0
    while True:
        result = workspace.read_reference("R1", entry_offset=offset, entry_limit=1)
        assert result["content_offset"] == offset
        assert result["content_total_lines"] == 45
        assert result["content_limit"] == 20
        parts.append(result["content"])
        offset = result["next_offset"]
        if offset is None:
            break
    assert len(parts) == 3
    assert "".join(parts) == text


def test_formatted_reference_retains_original_entry_lookup():
    workspace = workspace_for(reference_with_presentation("## Operators\n\nA formatted view.\n"))
    original = workspace.read_reference_entry("R1", "RE1")
    assert original["entry"]["content"] == "** ^ 2 power"
    assert original["entry"]["citations"][0]["evidence_id"] == "E1"


def test_formatted_lookup_keeps_literal_source_syntax_and_code_whitespace():
    text = "## Syntax\n\n```text\n  ![image](file.png)\n  [image example]\n```\n"
    result = workspace_for(reference_with_presentation(text)).read_reference("R1")
    assert result["content"] == text


def test_source_preserved_reference_keeps_existing_entry_pagination():
    workspace = workspace_for(reference_with_presentation())
    result = workspace.read_reference("R1", entry_offset=0, entry_limit=1)
    assert "formatting_mode" not in result
    assert result["entry_offset"] == 0
    assert result["entry_total"] == 1
    assert result["entries"][0]["entry_id"] == "RE1"


def test_formatted_reference_bounds_invalid_offsets_and_limits():
    workspace = workspace_for(reference_with_presentation("one\ntwo\n"))
    first = workspace.read_reference("R1", entry_offset=-10, entry_limit=0)
    assert first["content_offset"] == 0
    assert first["content_limit"] == 20
    assert first["content"] == "one\ntwo\n"
    end = workspace.read_reference("R1", entry_offset=100, entry_limit=500)
    assert end["content_offset"] == 2
    assert end["content_limit"] == 600
    assert end["content"] == ""
    assert end["next_offset"] is None
