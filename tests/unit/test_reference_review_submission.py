from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.knowledge import (
    KnowledgeToolExecutor,
    KnowledgeWorkspace,
    _build_reference_catalogs,
    validate_references,
)
from app.models import (
    CorpusSnapshot,
    EvidenceExcerpt,
    EvidenceNode,
    EvidenceSourceSpan,
    EvidenceTree,
    SourceResult,
    TargetProfile,
)
from app.storage import RunStore


def _workspace(
    tmp_path: Path,
    *,
    catalog_count: int = 1,
    catalog_entry_count: int = 4,
    ordinary_count: int = 0,
    ordinary_padding_chars: int = 0,
    oversized: bool = False,
    capacity: int = 32,
) -> KnowledgeWorkspace:
    target = TargetProfile(
        software_name="ExampleCalc",
        target_version="1.0",
        platform="General",
        ui_locale="English",
    )
    evidence: list[EvidenceExcerpt] = []
    for index in range(catalog_count):
        padding = "x" * (11_200 // catalog_entry_count) if oversized else ""
        content = "\n".join(
            f"r.topic{index}_{entry}\nDocumented operation {entry}. {padding}"
            for entry in range(catalog_entry_count)
        )
        evidence.append(_evidence(index, f"Command family {index} manual", content))
    for index in range(ordinary_count):
        content = (
            f"Valid syntax rule {index}: expression_{index}(x) accepts a numeric argument. "
            "Use parentheses to group operands; addition binds less tightly than multiplication."
        ) + " y" * (ordinary_padding_chars // 2)
        evidence.append(_evidence(catalog_count + index, f"Syntax rule {index}", content))
    nodes = [
        EvidenceNode(
            evidence_node_id=f"N{index + 2}",
            source_id=item.source_id,
            node_key=f"section-{index}",
            parent_node_id="N1",
            ordinal=index,
            node_type="reference",
            title=item.title,
            title_origin="source",
            source_refs=[
                EvidenceSourceSpan(
                    source_part_id=item.source_part_id,
                    text_artifact_id=item.text_artifact_id,
                    start_offset=0,
                    end_offset=len(item.content),
                    content_sha256=item.content_sha256,
                    locator=item.locator,
                )
            ],
            content_artifact_id=item.text_artifact_id,
            content_sha256=item.content_sha256,
            content=item.content,
            direct_char_count=len(item.content),
        )
        for index, item in enumerate(evidence)
    ]
    tree = EvidenceTree(
        evidence_tree_id="ET1",
        source_id="src_review",
        root_node_id="N1",
        nodes=[
            EvidenceNode(
                evidence_node_id="N1",
                source_id="src_review",
                node_key="root",
                child_node_ids=[item.evidence_node_id for item in nodes],
                ordinal=0,
                node_type="document",
                title="ExampleCalc manual",
                title_origin="source",
            ),
            *nodes,
        ],
        plan_artifact_id="art_review_plan",
        validation_artifact_id="art_review_validation",
    )
    corpus = CorpusSnapshot(
        corpus_snapshot_id="corpus_review",
        run_id="run_review",
        target_profile=target,
        source_results=[
            SourceResult(
                source_id="src_review",
                display_name="manual.html",
                modality="html",
                status="succeeded",
            )
        ],
        evidence=evidence,
        evidence_trees=[tree],
        evidence_count=len(evidence),
        source_part_count=len(evidence),
        source_status_counts={"succeeded": 1},
    )
    workspace = KnowledgeWorkspace(
        store=RunStore(tmp_path, "run_review"),
        settings=Settings(
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
            max_knowledge_candidates=capacity,
        ),
        target_profile=target,
        stage="reference_build",
        corpus=corpus,
        corpus_artifact_id="art_review_corpus",
    )
    # Keep independent authored catalogs independent in this review fixture.
    catalogs = [
        catalog
        for item in evidence[:catalog_count]
        for catalog in _build_reference_catalogs(corpus.corpus_snapshot_id, [item])
    ]
    assert len(catalogs) == catalog_count
    for catalog in catalogs:
        original_id = str(catalog["catalog_id"])
        catalog["catalog_id"] = workspace.store.canonical_id("C", original_id, original_id)
    workspace.reference_catalogs = catalogs
    workspace.reference_catalogs_by_id = {item["catalog_id"]: item for item in catalogs}
    return workspace


def _evidence(index: int, title: str, content: str) -> EvidenceExcerpt:
    return EvidenceExcerpt(
        evidence_id=f"evi_review_{index}",
        source_id="src_review",
        source_part_id=f"part_review_{index}",
        text_artifact_id=f"art_review_{index}",
        start_offset=0,
        end_offset=len(content),
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        kind="html_section",
        title=title,
        content=content,
        locator={"heading": title, "url": f"https://docs.example/section-{index}.html"},
        extraction_method="test",
    )


def _request(workspace: KnowledgeWorkspace) -> dict[str, Any]:
    manifest = workspace.prepare_reference_build()
    catalog_decisions = [
        {
            "catalog_id": item["catalog_id"],
            "disposition": "preserve",
            "title": item["suggested_title"],
            "description": "ExampleCalc command lookup. Consult to check available operations.",
        }
        for item in manifest["catalog_candidates"]
    ]
    return {
        "catalog_decisions": catalog_decisions,
        "ordinary_decisions": [
            {
                "candidate_id": item["candidate_id"],
                "disposition": "exclude",
            }
            for item in manifest["ordinary_candidates"]
        ],
        "partition_decisions": [
            {
                "subject_id": item["subject_id"],
                "decision": "keep_whole",
                "groups": [],
            }
            for item in manifest["partition_review_candidates"]
            if item["subject_id"] in {decision["catalog_id"] for decision in catalog_decisions}
        ],
        "status": "succeeded",
        "summary": "The Agent explicitly reviewed every supplied Reference candidate.",
    }


def _assert_unsubmitted(workspace: KnowledgeWorkspace) -> None:
    assert workspace.submitted_references == []
    assert workspace.submitted_catalog_ids == set()
    assert workspace.reference_catalog_review == {}
    assert workspace.reference_candidate_review == {}
    assert workspace.reference_partition_review == {}
    assert workspace.stage_result is None


def test_high_confidence_catalogs_remain_unsubmitted_candidates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    manifest = workspace.prepare_reference_build()

    assert manifest["mode"] == "python_candidates_agent_review"
    assert manifest["high_confidence_catalog_count"] == 1
    assert len(manifest["catalog_candidates"]) == 1
    candidate = manifest["catalog_candidates"][0]
    assert candidate["confidence_hint"] == "high"
    assert candidate["structure_proof"]["confidence"] == 1.0
    assert "automatic" not in candidate["structure_proof"]
    assert candidate["samples"]
    assert "automatic_catalogs" not in manifest
    _assert_unsubmitted(workspace)
    assert workspace.prepare_reference_build() is manifest


def test_agent_can_exclude_a_high_confidence_oversized_catalog(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, oversized=True)
    request = _request(workspace)
    request["catalog_decisions"][0] = {
        "catalog_id": request["catalog_decisions"][0]["catalog_id"],
        "disposition": "exclude",
    }
    request["partition_decisions"] = []

    workspace.submit_reference_review(**request)

    assert workspace.submitted_references == []
    assert workspace.inspect_reference_coverage()["unreviewed_catalog_ids"] == []
    assert len(workspace.inspect_reference_coverage()["excluded"]) == 1
    assert workspace.reference_partition_review == {}


@pytest.mark.parametrize("hint", ["high", "ambiguous"])
@pytest.mark.parametrize("field", ["title", "description"])
def test_preserved_catalog_requires_agent_metadata(
    tmp_path: Path, hint: str, field: str
) -> None:
    workspace = _workspace(tmp_path)
    if hint == "ambiguous":
        workspace.reference_catalogs[0]["structure"] = "repeated_field_catalog"
    request = _request(workspace)
    assert workspace.prepare_reference_build()["catalog_candidates"][0]["confidence_hint"] == hint
    request["catalog_decisions"][0].pop(field)

    with pytest.raises(ValueError, match=field):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)


@pytest.mark.parametrize("decision_type", ["catalog_decisions", "ordinary_decisions"])
def test_every_supplied_candidate_requires_an_explicit_decision(
    tmp_path: Path, decision_type: str
) -> None:
    workspace = _workspace(tmp_path, ordinary_count=1)
    request = _request(workspace)
    assert request[decision_type]
    request[decision_type].pop()

    with pytest.raises(ValueError, match="Missing|missing"):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)


def test_ordinary_preserve_requires_agent_description(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=1)
    request = _request(workspace)
    request["ordinary_decisions"][0].update(
        disposition="preserve", title="ExampleCalc expression syntax"
    )

    with pytest.raises(ValueError, match="description"):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)


def test_submission_separates_agent_description_from_exact_source_entries(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    request = _request(workspace)
    source_entries = workspace.reference_catalogs[0]["catalog_entries"]

    workspace.submit_reference_review(**request)

    reference = workspace.submitted_references[0]
    assert reference.description == request["catalog_decisions"][0]["description"]
    assert reference.content == ""
    assert [entry.content for entry in reference.entries] == [entry["content"] for entry in source_entries]
    assert [entry.title for entry in reference.entries] == [entry["title"] for entry in source_entries]
    validated, validation = validate_references([reference], workspace.evidence_by_id)
    assert validated and validation[0].accepted


def test_ordinary_preserve_retains_the_whole_source_region(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=1)
    request = _request(workspace)
    description = "ExampleCalc expression syntax and operator precedence. Consult when checking valid expressions."
    request["ordinary_decisions"][0].update(
        disposition="preserve",
        title="ExampleCalc expression syntax",
        description=description,
    )

    workspace.submit_reference_review(**request)

    reference = workspace.submitted_references[0]
    assert reference.description == description
    assert reference.content == ""
    assert [entry.content for entry in reference.entries] == [workspace.corpus.evidence[0].content]
    validated, validation = validate_references([reference], workspace.evidence_by_id)
    assert validated and validation[0].accepted


def test_more_than_64_catalogs_can_be_reviewed_without_consuming_final_capacity(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=65, capacity=1)
    request = _request(workspace)
    assert len(request["catalog_decisions"]) == 65
    for decision in request["catalog_decisions"]:
        decision.update(disposition="exclude")

    workspace.submit_reference_review(**request)

    assert workspace.submitted_references == []
    coverage = workspace.inspect_reference_coverage()
    assert len(coverage["excluded"]) == 65
    assert coverage["unreviewed_catalog_ids"] == []


def test_preserved_oversized_catalog_requires_partition_review(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, oversized=True)
    request = _request(workspace)
    assert request["partition_decisions"]
    request["partition_decisions"] = []

    with pytest.raises(ValueError, match="partition|Partition"):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)


def _split_request(request: dict[str, Any]) -> None:
    request["partition_decisions"][0].update(
        decision="split",
        groups=[
            {
                "title": "ExampleCalc first command family",
                "description": "ExampleCalc first command family. Consult to identify these documented commands.",
                "entry_indices": [0, 1],
            },
            {
                "title": "ExampleCalc second command family",
                "description": "ExampleCalc second command family. Consult to check these documented capabilities.",
                "entry_indices": [2, 3],
            },
        ],
    )


@pytest.mark.parametrize("partition", ["keep_whole", "split"])
@pytest.mark.parametrize("legacy_payload", [False, True])
def test_review_submits_without_rationales_and_drops_legacy_fields(
    tmp_path: Path, partition: str, legacy_payload: bool,
) -> None:
    workspace = _workspace(tmp_path, catalog_count=2, ordinary_count=1, oversized=True)
    request = _request(workspace)
    excluded_id = request["catalog_decisions"][1]["catalog_id"]
    request["catalog_decisions"][1] = {"catalog_id": excluded_id, "disposition": "exclude"}
    request["partition_decisions"] = [
        decision for decision in request["partition_decisions"]
        if decision["subject_id"] != excluded_id
    ]
    if partition == "split":
        _split_request(request)
    if legacy_payload:
        for field in ("catalog_decisions", "ordinary_decisions", "partition_decisions"):
            for decision in request[field]:
                decision["rationale"] = "Historical review explanation."
    original_request = deepcopy(request)

    result = KnowledgeToolExecutor(workspace, "reference_test_agent").call(
        "submit_reference_review", request,
    )

    assert result["ok"] and result["accepted"]
    assert request == original_request
    assert workspace.stage_result is not None
    assert workspace.inspect_reference_coverage()["excluded"] == [{"catalog_id": excluded_id}]
    assert len(workspace.submitted_references) == (2 if partition == "split" else 1)
    assert set(workspace.reference_candidate_review) == {
        decision["candidate_id"] for decision in request["ordinary_decisions"]
    }
    assert len(workspace.reference_partition_review) == 1
    saved_review = {
        "catalogs": workspace.reference_catalog_review,
        "ordinary": workspace.reference_candidate_review,
        "partitions": workspace.reference_partition_review,
        "stage_result": workspace.stage_result,
        "tool_result": result,
    }
    assert "rationale" not in json.dumps(saved_review)


def test_catalog_coverage_supports_new_and_legacy_reviews(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=2)
    covered_id, excluded_id = [catalog["catalog_id"] for catalog in workspace.reference_catalogs]
    workspace.submit_reference_catalog(
        catalog_id=covered_id, title="ExampleCalc command lookup",
        description="ExampleCalc command lookup. Consult to check available operations.",
    )

    result = workspace.review_reference_coverage([
        {"catalog_id": covered_id, "disposition": "covered"},
        {"catalog_id": excluded_id, "disposition": "excluded"},
    ])

    assert result["coverage"]["excluded"] == [{"catalog_id": excluded_id}]
    assert workspace.reference_catalog_review[excluded_id] == {"disposition": "excluded"}
    # Historical saved metadata remains readable without rewriting it.
    workspace.reference_catalog_review[excluded_id]["rationale"] = "Historical exclusion explanation."
    original_review = deepcopy(workspace.reference_catalog_review)
    assert workspace.inspect_reference_coverage()["excluded"] == [{"catalog_id": excluded_id}]
    assert workspace.reference_catalog_review == original_review


def test_failed_partition_rolls_back_before_successful_retry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, oversized=True)
    request = _request(workspace)
    _split_request(request)
    request["partition_decisions"][0]["groups"][1]["entry_indices"] = [1, 2, 3]

    with pytest.raises(ValueError, match="every entry|exactly once"):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)
    request["partition_decisions"][0]["groups"][1]["entry_indices"] = [2, 3]
    workspace.submit_reference_review(**request)
    assert len(workspace.submitted_references) == 2
    assert [reference.description for reference in workspace.submitted_references] == [
        group["description"] for group in request["partition_decisions"][0]["groups"]
    ]
    assert all(reference.content == "" for reference in workspace.submitted_references)
    assert [
        entry.content for reference in workspace.submitted_references for entry in reference.entries
    ] == [entry["content"] for entry in workspace.reference_catalogs[0]["catalog_entries"]]


def test_split_description_cannot_fall_back_to_generated_text(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, oversized=True)
    request = _request(workspace)
    _split_request(request)
    request["partition_decisions"][0]["groups"][1]["description"] = "  "

    with pytest.raises(ValueError, match="description"):
        workspace.submit_reference_review(**request)

    _assert_unsubmitted(workspace)


@pytest.mark.parametrize("capacity", [1, 2])
def test_final_capacity_counts_split_references_at_the_boundary(
    tmp_path: Path, capacity: int
) -> None:
    workspace = _workspace(tmp_path, oversized=True, capacity=capacity)
    request = _request(workspace)
    _split_request(request)

    if capacity == 1:
        with pytest.raises(ValueError, match="capacity|slots"):
            workspace.submit_reference_review(**request)
        _assert_unsubmitted(workspace)
    else:
        workspace.submit_reference_review(**request)
        assert len(workspace.submitted_references) == 2


def test_catalog_capacity_failure_leaves_no_partial_submission(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=2, capacity=1)

    with pytest.raises(ValueError, match="capacity|slots"):
        workspace.submit_reference_review(**_request(workspace))

    _assert_unsubmitted(workspace)


def test_ordinary_candidates_are_not_truncated_to_six(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=8)
    manifest = workspace.prepare_reference_build()

    assert manifest["ordinary_candidate_count"] == 8
    assert len(manifest["ordinary_candidates"]) == 8
    workspace.submit_reference_review(**_request(workspace))
    assert workspace.stage_result is not None


def test_inspectors_accept_high_confidence_candidates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate = workspace.prepare_reference_build()["catalog_candidates"][0]
    executor = KnowledgeToolExecutor(workspace, "reference_test_agent")

    single = executor.call("inspect_reference_catalog", {"catalog_id": candidate["catalog_id"]})
    batch = executor.call("inspect_reference_catalog_batch", {"catalog_ids": [candidate["catalog_id"]]})

    assert single["ok"] and single["entries"]
    assert batch["ok"] and batch["catalogs"]


def test_inspect_reference_candidates_batch_reads_in_order_and_dedupes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=4)
    manifest = workspace.prepare_reference_build()
    executor = KnowledgeToolExecutor(workspace, "reference_test_agent")
    ids = [item["candidate_id"] for item in manifest["ordinary_candidates"][:3]]

    result = executor.call(
        "inspect_reference_candidates", {"candidate_ids": [ids[0], ids[1], ids[0], ids[2]], "limit": 2}
    )

    assert result["ok"]
    assert [item["candidate_id"] for item in result["candidates"]] == ids
    assert result["limit"] == 2
    for item in result["candidates"]:
        assert len(item["evidence"]) <= 2
        assert item["offset"] == 0


def test_inspect_reference_candidates_reject_whole_batch_naming_failures(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=2)
    manifest = workspace.prepare_reference_build()
    executor = KnowledgeToolExecutor(workspace, "reference_test_agent")
    good = manifest["ordinary_candidates"][0]["candidate_id"]

    result = executor.call(
        "inspect_reference_candidates", {"candidate_ids": [good, "N999", "", "N888"]}
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_candidate_batch"
    assert "''" in result["message"]

    outside = executor.call(
        "inspect_reference_candidates", {"candidate_ids": [good, "N999", "N888"]}
    )
    assert outside["ok"] is False
    assert outside["error"] == "candidates_not_reviewable"
    assert outside["candidate_ids"] == ["N999", "N888"]
    assert good not in outside["message"]

    assert executor.call("inspect_reference_candidates", {"candidate_ids": []})["ok"] is False
    assert executor.call("inspect_reference_candidates", {"candidate_ids": "N1"})["ok"] is False
    oversized = executor.call(
        "inspect_reference_candidates", {"candidate_ids": [f"N{i}" for i in range(9)]}
    )
    assert oversized["ok"] is False


def test_review_can_inspect_more_than_three_candidates_and_ten_evidence_spans(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_count=0, ordinary_count=12)
    manifest = workspace.prepare_reference_build()
    executor = KnowledgeToolExecutor(workspace, "reference_test_agent")

    for candidate in manifest["ordinary_candidates"]:
        result = executor.call("inspect_reference_candidate", {"candidate_id": candidate["candidate_id"]})
        assert result["ok"] and not result.get("skipped")
    for evidence in workspace.corpus.evidence:
        result = executor.call("read_evidence", {"evidence_id": evidence.evidence_id})
        assert result["ok"] and not result.get("skipped")


def test_catalog_inspection_pages_beyond_two_calls_with_absolute_entry_indices(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, catalog_entry_count=20, oversized=True)
    manifest = workspace.prepare_reference_build()
    catalog_id = manifest["catalog_candidates"][0]["catalog_id"]
    executor = KnowledgeToolExecutor(workspace, "reference_test_agent")
    returned_entries = []

    for offset in (0, 8, 16):
        page = executor.call(
            "inspect_reference_catalog",
            {"catalog_id": catalog_id, "offset": offset, "limit": 999},
        )
        assert page["ok"]
        assert page["limit"] == 8
        assert [entry["entry_index"] for entry in page["entries"]] == list(
            range(offset, min(offset + 8, 20))
        )
        returned_entries.extend(page["entries"])

    assert [entry["entry_index"] for entry in returned_entries] == list(range(20))
    partition = next(item for item in manifest["partition_review_candidates"] if item["subject_id"] == catalog_id)
    assert [entry["entry_index"] for entry in partition["outline"]] == list(range(20))
    assert [entry["source_evidence_id"] for entry in partition["outline"]] == [
        entry["evidence_id"] for entry in returned_entries
    ]


def test_mixed_ordinary_unit_uses_the_same_filtered_evidence_through_submission(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path, ordinary_count=1, oversized=True, ordinary_padding_chars=11_000
    )
    catalog = workspace.reference_catalogs[0]
    structural_evidence_id = catalog["evidence_ids"][0]
    catalog.update(structure="repeated_field_catalog", catalog_entries=[], entry_count=1)
    structural_unit, ordinary_unit = workspace.reference_base_units
    ordinary_evidence_id = ordinary_unit["evidence_ids"][0]
    mixed_unit = {
        **ordinary_unit,
        "evidence_ids": [structural_evidence_id, ordinary_evidence_id],
        "source_parts": structural_unit["source_parts"] + ordinary_unit["source_parts"],
        "source_part_count": 2,
        "direct_char_count": structural_unit["direct_char_count"] + ordinary_unit["direct_char_count"],
    }
    candidate_id = mixed_unit["evidence_node_id"]
    workspace.reference_base_units = [mixed_unit]
    workspace.reference_base_units_by_node_id = {candidate_id: mixed_unit}
    workspace.reference_units = [mixed_unit]
    workspace.reference_units_by_id = {mixed_unit["unit_id"]: mixed_unit}
    workspace._task_node_ids_by_evidence_id = {
        structural_evidence_id: [candidate_id], ordinary_evidence_id: [candidate_id]
    }

    manifest = workspace.prepare_reference_build()
    assert len(manifest["ordinary_candidates"]) == 1
    ordinary_candidate = manifest["ordinary_candidates"][0]
    assert ordinary_candidate["evidence_count"] == 1
    assert ordinary_candidate["evidence_id_samples"] == [ordinary_evidence_id]
    assert "Valid syntax rule" in ordinary_candidate["representative_excerpt"]
    assert "r.topic" not in ordinary_candidate["representative_excerpt"]
    inspection = workspace.inspect_reference_candidate(candidate_id)
    assert inspection["evidence_count"] == 1
    assert [item["evidence_id"] for item in inspection["evidence"]] == [ordinary_evidence_id]
    rows = workspace._partition_rows_for_unit(mixed_unit)
    assert [item["source_evidence_id"] for item in rows] == [ordinary_evidence_id]
    partition = next(item for item in manifest["partition_review_candidates"] if item["subject_id"] == candidate_id)
    assert partition["entry_count"] == 1
    assert [item["source_evidence_id"] for item in partition["outline"]] == [ordinary_evidence_id]
    request = _request(workspace)
    request["catalog_decisions"][0].update(disposition="exclude")
    request["ordinary_decisions"][0].update(
        disposition="preserve",
        title="ExampleCalc expression syntax",
        description="ExampleCalc expression rules and precedence. Consult when checking valid expressions.",
    )
    request["partition_decisions"] = [{
        "subject_id": candidate_id,
        "decision": "keep_whole",
        "groups": [],
    }]

    workspace.submit_reference_review(**request)

    assert len(workspace.submitted_references) == 1
    reference = workspace.submitted_references[0]
    assert [entry.source_evidence_id for entry in reference.entries] == [ordinary_evidence_id]
    assert [entry.content for entry in reference.entries] == [workspace.evidence_by_id[ordinary_evidence_id].content]
    assert reference.description == request["ordinary_decisions"][0]["description"]
    assert structural_evidence_id not in {citation.evidence_id for citation in reference.citations}


@pytest.mark.parametrize("kind", ["catalog", "ordinary", "split"])
@pytest.mark.parametrize("character", ["x", "知"])
def test_description_limit_rejects_atomically_and_preserves_source(
    tmp_path: Path, kind: str, character: str,
) -> None:
    workspace = _workspace(
        tmp_path, catalog_count=0 if kind == "ordinary" else 1,
        ordinary_count=1 if kind == "ordinary" else 0, oversized=kind == "split",
    )
    request = _request(workspace)
    if kind == "ordinary":
        decision = request["ordinary_decisions"][0]
        decision.update(disposition="preserve", title="ExampleCalc expression syntax")
    elif kind == "split":
        _split_request(request)
        decision = request["partition_decisions"][0]["groups"][0]
    else:
        decision = request["catalog_decisions"][0]
    expected_content = (
        [workspace.corpus.evidence[0].content] if kind == "ordinary"
        else [entry["content"] for entry in workspace.reference_catalogs[0]["catalog_entries"]]
    )
    decision["description"] = character * 121
    with pytest.raises(ValueError, match="1 to 120 characters"):
        workspace.submit_reference_review(**request)
    _assert_unsubmitted(workspace)
    decision["description"] = character * 120
    workspace.submit_reference_review(**request)
    assert workspace.submitted_references[0].description == character * 120
    assert [
        entry.content for reference in workspace.submitted_references for entry in reference.entries
    ] == expected_content
