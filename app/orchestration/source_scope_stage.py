"""Choose knowledge scope from reviewed Evidence, preserving the complete corpus."""
from __future__ import annotations

import hashlib
from typing import Any

from ..models import EvidencePlan, EvidencePlanNode, EvidenceSourceRef, SourceManifest, SourcePart, SourceResult
from ..storage import RunStore


def receipt_artifact(store: RunStore, stage: str, pointer: str) -> str | None:
    # Receipts are authoritative if the process stopped before updating a pointer.
    for commit in reversed(store.list_receipt_commits()):
        receipt = commit.get("receipt", {})
        if receipt.get("stage") == stage:
            outputs = receipt.get("accepted_output_artifact_ids", [])
            return str(outputs[0]) if outputs else None
    value = store.read_state().get(pointer)
    return str(value) if value else None


def build_outline_preview(store: RunStore, results: list[SourceResult]) -> str:
    """Project the accepted, fully parsed tree; never guess HTML/PDF boundaries."""
    active = [item for item in results if item.status in {"succeeded", "degraded"}]
    if len(active) != 1 or active[0].evidence_tree is None:
        raise ValueError("Chapter selection requires exactly one source with a reviewed Evidence tree")
    result = active[0]
    manifest, tree = result.manifest, result.evidence_tree
    by_id = {node.evidence_node_id: node for node in tree.nodes}
    sections: list[dict[str, Any]] = []
    for node_id in by_id[tree.root_node_id].child_node_ids:
        node = by_id[node_id]
        sections.append({
            "section_id": node_id, "title": node.title, "depth": 0,
            "parent_section_id": node.parent_node_id,
            "direct_char_count": node.direct_char_count,
            "descendant_char_count": node.descendant_char_count,
            "node_type": node.node_type,
        })
    payload = {
        "schema_version": "source-outline.v2",
        "source_id": manifest.source_id,
        "modality": manifest.primary_modality,
        "title": manifest.display_name,
        "raw_artifact_id": manifest.raw_artifact_id,
        "raw_sha256": store.artifact_record(manifest.raw_artifact_id).sha256,
        "reviewed_corpus_artifact_id": result.corpus_artifact_id,
        "evidence_tree_id": tree.evidence_tree_id,
        "outline_available": bool(sections), "sections": sections,
        "warnings": list(dict.fromkeys([*result.warnings, *result.gaps])),
        "coverage": result.coverage.model_dump(mode="json"),
        "acquisition_coverage": manifest.source_metadata.get("acquisition_coverage", {}),
        "structure_review_complete": True,
    }
    record = store.put_json_artifact(payload, kind="source_outline_preview", source_id=manifest.source_id)
    return record.artifact_id


def selectable_sections(outline: dict[str, Any]) -> list[dict[str, Any]]:
    """Also hide descendants from older, already persisted v2 outlines."""
    return [item for item in outline.get("sections", []) if item.get("depth", 0) == 0]


def validate_selection(outline: dict[str, Any], mode: str, section_ids: list[str],
                       *, allow_committed_descendants: bool = False) -> list[str]:
    if mode not in {"full", "selected"}:
        raise ValueError("Choose full-document extraction or selected chapters")
    if mode == "full":
        if section_ids:
            raise ValueError("Full-document extraction cannot include selected chapter IDs")
        return []
    if not outline.get("outline_available"):
        raise ValueError("This source has no reliable chapter outline; choose full-document extraction")
    if not section_ids:
        raise ValueError("Select at least one chapter")
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Chapter IDs must be unique")
    sections = outline.get("sections", []) if allow_committed_descendants else selectable_sections(outline)
    ordered = [item["section_id"] for item in sections]
    if set(section_ids) - set(ordered):
        raise ValueError("Unknown chapter or non-first-level chapter in this outline preview")
    return [section_id for section_id in ordered if section_id in section_ids]


def selected_scope(store: RunStore) -> dict[str, Any] | None:
    artifact_id = receipt_artifact(store, "scope_selection", "scope_selection_artifact_id")
    return store.read_json_artifact(artifact_id) if artifact_id else None


def apply_scope(store: RunStore, manifests: list[SourceManifest], selection: dict[str, Any]) -> None:
    """Legacy v1 compatibility only; new runs never filter before extraction."""
    outline_id = selection["outline_artifact_id"]
    outline = store.read_json_artifact(outline_id)
    section_ids = validate_selection(outline, selection["mode"], selection["section_ids"])
    manifest = next((item for item in manifests if item.source_id == outline["source_id"]), None)
    if manifest is None or manifest.raw_artifact_id != outline["raw_artifact_id"]:
        raise RuntimeError("The source changed after its outline preview; start a new Run")
    path = store.artifact_path(manifest.raw_artifact_id)
    if hashlib.sha256(path.read_bytes()).hexdigest() != outline["raw_sha256"]:
        raise RuntimeError("The original material no longer matches its outline preview")
    manifest.source_metadata["extraction_scope"] = {
        "mode": selection["mode"], "section_ids": section_ids,
        "sections": outline.get("sections", []), "outline_artifact_id": outline_id,
        "max_child_pages": outline["max_child_pages"],
        "page_budget_scope": "child_pages_only",
    }
    store.write_source_json(manifest.source_id, "manifest.json", manifest.model_dump(mode="json"))


def project_reviewed_scope(store: RunStore, results: list[SourceResult], selection: dict[str, Any],
                           *, settings: Any, target: Any) -> list[SourceResult]:
    """Copy exact selected spans to a new corpus, including structural ancestors only.

    Immutable reviewed corpora remain untouched. Mutable Source files are active
    projections; retries rebuild them from the accepted full corpus.
    """
    from ..acquisition import AcquisitionService
    from ..evidence_planning import EvidencePlanEngine
    from ..source_tools.workspace import SourceWorkspace
    from ..toolkit import ToolExecutor

    outline = store.read_json_artifact(selection["outline_artifact_id"])
    ids = validate_selection(outline, selection["mode"], selection["section_ids"],
                             allow_committed_descendants=selected_scope(store) == selection)
    if outline.get("schema_version") != "source-outline.v2":
        raise RuntimeError("A reviewed Evidence outline is required before scope projection")
    result = next(item for item in results if item.source_id == outline["source_id"])
    if (result.corpus_artifact_id != outline["reviewed_corpus_artifact_id"]
            or result.evidence_tree.evidence_tree_id != outline["evidence_tree_id"]):
        raise RuntimeError("The reviewed Evidence changed after chapter selection")
    # Always restore the accepted parts from their immutable full corpus. A prior
    # attempt may have stopped while replacing mutable Source projections.
    corpus_record = store.artifact_record(result.corpus_artifact_id)
    corpus_bytes = store.artifact_path(result.corpus_artifact_id).read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != corpus_record.sha256:
        raise RuntimeError("The reviewed corpus failed its immutable content hash")
    full = store.read_json_artifact(result.corpus_artifact_id)
    original_parts = {item["part_id"]: SourcePart.model_validate(item) for item in full["parts"]}
    tree = result.evidence_tree
    by_id = {node.evidence_node_id: node for node in tree.nodes}
    selected: set[str] = set()
    pending = list(ids) if selection["mode"] == "selected" else [tree.root_node_id]
    while pending:
        node_id = pending.pop()
        if node_id not in selected:
            selected.add(node_id)
            pending.extend(by_id[node_id].child_node_ids)
    retained = set(selected)
    for node_id in selected:
        parent = by_id[node_id].parent_node_id
        while parent and parent not in retained:
            retained.add(parent)
            parent = by_id[parent].parent_node_id
    manifest = result.manifest.model_copy(deep=True)
    manifest.source_metadata["extraction_scope"] = {
        "stage": "after_source_review", "mode": selection["mode"], "section_ids": ids,
        "outline_artifact_id": selection["outline_artifact_id"],
        "reviewed_corpus_artifact_id": result.corpus_artifact_id,
        "reviewed_evidence_tree_id": tree.evidence_tree_id,
    }
    workspace = SourceWorkspace(store, settings, AcquisitionService(store, settings), target, manifest)
    workspace.gaps = list(result.gaps)
    workspace.warnings = list(result.warnings)
    workspace.inventory_completed = True
    plan_nodes = []
    # Sort by original part order rather than tree order (a reviewer may move nodes).
    staged_parts = []
    part_order = {part_id: index for index, part_id in enumerate(original_parts)}
    for node in tree.nodes:
        if node.evidence_node_id not in retained:
            continue
        refs = []
        for span in node.source_refs if node.evidence_node_id in selected else []:
            original = original_parts[span.source_part_id]
            text = store.artifact_path(span.text_artifact_id).read_bytes().decode("utf-8")
            content = text[span.start_offset:span.end_offset]
            if hashlib.sha256(content.encode("utf-8")).hexdigest() != span.content_sha256:
                raise RuntimeError("Reviewed source span no longer matches its immutable text")
            digest = hashlib.sha256(
                f"{selection['outline_artifact_id']}:{span.source_part_id}:{span.start_offset}:{span.end_offset}".encode()
            ).hexdigest()[:24]
            projected_id = f"part_{digest}"
            images = []
            for image in original.metadata.get("image_refs", []):
                if span.start_offset <= int(image.get("anchor_offset", 0)) < span.end_offset:
                    adjusted = dict(image, anchor_offset=int(image.get("anchor_offset", 0)) - span.start_offset)
                    if image["source_part_id"] == original.part_id:
                        adjusted["source_part_id"] = projected_id
                    images.append(adjusted)
                    if image["source_part_id"] != original.part_id:
                        image_part = original_parts[image["source_part_id"]].model_copy(deep=True)
                        image_part.parent_part_id = None
                        image_part.text_artifact_id = None
                        image_part.char_count = 0
                        image_part.metadata = {}
                        workspace.parts[image_part.part_id] = image_part
            part = SourcePart(
                part_id=projected_id, source_id=result.source_id, kind=original.kind,
                label=node.title, locator=dict(span.locator), ordinal=original.ordinal,
                metadata={"scope_projection": True, "image_refs": images,
                          "images": {str(index): {"status": "saved", "artifact_id": image["artifact_id"],
                                                  "media_type": image["media_type"]} for index, image in enumerate(images)},
                          "origin_source_part_id": original.part_id,
                          "origin_text_artifact_id": span.text_artifact_id,
                          "origin_start_offset": span.start_offset, "origin_end_offset": span.end_offset},
            )
            staged_parts.append((part_order[original.part_id], span.start_offset, part, content))
            refs.append(EvidenceSourceRef(source_part_id=part.part_id))
        plan_nodes.append(EvidencePlanNode(
            node_key=node.node_key, parent_node_key=(by_id[node.parent_node_id].node_key if node.parent_node_id else None),
            node_type=node.node_type, title=node.title, title_origin=node.title_origin, source_refs=refs,
        ))
    for _, _, part, text in sorted(staged_parts, key=lambda item: item[:2]):
        workspace.add_part(part, text)
        workspace.inventoried_parts.add(part.part_id)
    if not staged_parts:
        raise ValueError("The selected chapters contain no source-grounded text")
    plan = EvidencePlan(source_id=result.source_id, nodes=plan_nodes)
    artifacts = EvidencePlanEngine(workspace).materialize(plan)
    ToolExecutor(workspace, "scope_projection")._commit_materialized_evidence(
        plan, artifacts, summary=result.summary, gaps=workspace.gaps, warnings=workspace.warnings,
        accepted_event="reviewed_scope_materialized",
    )
    return [workspace.final_result if item.source_id == result.source_id else item for item in results]


def scope_payload(store: RunStore) -> dict[str, Any] | None:
    artifact_id = receipt_artifact(store, "scope_preview", "source_outline_artifact_id")
    if not artifact_id:
        return None
    outline = store.read_json_artifact(artifact_id)
    selection = selected_scope(store)
    if selection and selection["outline_artifact_id"] != artifact_id:
        selection = None
    return {
        **outline, "sections": selectable_sections(outline), "outline_artifact_id": artifact_id,
        "selection": ({"mode": selection["mode"], "section_ids": selection["section_ids"]}
                      if selection else None),
    }
