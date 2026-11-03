from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.models import ReferenceCandidate, ReferenceDraft


def _reference_payload(**changes: object) -> dict:
    payload = {
        "title": "Documented commands",
        "description": "ExampleApp command reference. Consult to check names and arguments.",
        "content": "The reset command accepts --force to suppress the documented prompt.",
        "citations": [{"evidence_id": "E1", "quote": "reset accepts --force"}],
    }
    payload.update(changes)
    return payload


def _saved_payload(**changes: object) -> dict:
    return _reference_payload(
        reference_id="R1", corpus_snapshot_id="corpus_saved", **changes
    )


def test_new_reference_requires_an_independent_description() -> None:
    payload = _reference_payload()
    del payload["description"]
    with pytest.raises(ValidationError, match="description"):
        ReferenceDraft.model_validate(payload)


@pytest.mark.parametrize("description", ["", "   ", "x" * 121, "x" * 1_025])
def test_new_reference_rejects_invalid_description(description: str) -> None:
    with pytest.raises(ValidationError, match="description"):
        ReferenceDraft.model_validate(_reference_payload(description=description))


def test_catalog_knowledge_can_live_in_entries_without_intro_content() -> None:
    payload = _reference_payload(reference_mode="source_catalog", entries=[{
        "entry_id": "RE1",
        "title": "reset",
        "content": "reset --force suppresses the documented prompt.",
        "source_evidence_id": "E1",
        "citations": [{"evidence_id": "E1", "quote": "reset --force"}],
    }])
    del payload["content"]
    draft = ReferenceDraft.model_validate(payload)
    assert draft.content == ""
    assert draft.entries[0].content.startswith("reset --force")


@pytest.mark.parametrize("content", ["", " \n "])
def test_summary_reference_requires_knowledge_content(content: str) -> None:
    with pytest.raises(ValidationError, match="nonempty knowledge content"):
        ReferenceDraft.model_validate(_reference_payload(content=content))


def test_legacy_catalog_reads_description_from_old_intro_without_mutation() -> None:
    payload = _saved_payload(reference_mode="source_catalog", content="Exact source-authored command and argument mappings.")
    del payload["description"]
    before = deepcopy(payload)
    reference = ReferenceCandidate.model_validate(payload)
    assert reference.description == payload["content"]
    assert reference.content == ""
    assert payload == before


def test_legacy_summary_preserves_knowledge_content_and_bounds_metadata() -> None:
    payload = _saved_payload(content="Syntax and valid command arguments. " * 150)
    del payload["description"]
    before = deepcopy(payload)
    reference = ReferenceCandidate.model_validate(payload)
    assert reference.content == payload["content"]
    assert reference.description.startswith("Documented commands: Syntax")
    assert len(reference.description) == 1_024
    assert payload == before


def test_legacy_long_catalog_intro_keeps_full_text_outside_bounded_metadata() -> None:
    payload = _saved_payload(
        reference_mode="source_catalog", content="Documented command lookup scope. " * 100
    )
    del payload["description"]
    before = deepcopy(payload)
    reference = ReferenceCandidate.model_validate(payload)
    assert reference.description == payload["content"][:1_024]
    assert reference.content == payload["content"]
    assert payload == before


def test_saved_reference_does_not_replace_an_explicit_empty_description() -> None:
    with pytest.raises(ValidationError, match="description"):
        ReferenceCandidate.model_validate(_saved_payload(description=""))


def test_current_catalog_retains_real_body_separate_from_description() -> None:
    payload = _saved_payload(reference_mode="source_catalog")
    reference = ReferenceCandidate.model_validate(payload)
    assert reference.description == payload["description"]
    assert reference.content == payload["content"]


@pytest.mark.parametrize("model", [ReferenceDraft, ReferenceCandidate])
def test_reference_models_drop_retired_fields_without_mutating_saved_payload(model) -> None:
    payload = _saved_payload(caveats=["Historical metadata; source restrictions remain in content."])
    before = deepcopy(payload)
    reference = model.model_validate(payload)
    assert "caveats" not in model.model_json_schema()["properties"]
    assert "caveats" not in reference.model_dump(mode="json")
    assert payload == before



@pytest.mark.parametrize("character", ["x", "知"])
@pytest.mark.parametrize("length", [100, 101, 120])
def test_new_reference_accepts_description_with_validation_headroom(character: str, length: int) -> None:
    draft = ReferenceDraft.model_validate(_reference_payload(description=character * length))
    assert draft.description == character * length


def test_saved_reference_preserves_a_preexisting_long_description() -> None:
    payload = _saved_payload(description=("Historical lookup description. " * 20).strip())
    before = deepcopy(payload)
    reference = ReferenceCandidate.model_validate(payload)
    assert reference.description == payload["description"]
    assert payload == before
