"""Exercise PDF cleanup instructions through the existing review/tool contract."""
from __future__ import annotations

from collections import Counter

import pytest

from app.acquisition import AcquisitionService
from app.capabilities import CapabilityGateway, CapabilityGatewayServer
from app.capabilities.gateway import CapabilityInvocation
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.models import (
    EvidencePlan,
    EvidencePlanNode,
    EvidenceSourceRef,
    ModelRuntimeConfig,
    SourceManifest,
    SourcePart,
    TargetProfile,
)
from app.orchestration.source_stage import (
    PDF_SOURCE_REVIEW_MAX_TOOL_CALLS,
    PDF_SOURCE_REVIEW_PROMPT_VERSION,
    PreparedSource,
    SourceReviewService,
)
from app.runtime import AgentRuntime, AgentTurnResult
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


class CleanupRuntime(AgentRuntime):
    """Script structural decisions, while using real capabilities and validation."""

    def __init__(self, gateway, operations):
        self.gateway = gateway
        self.operations = operations
        self.invocation = None

    async def run(self, invocation):
        self.invocation = invocation
        capability = invocation.metadata["capability"]
        call_index = 0

        def call(name, arguments):
            nonlocal call_index
            call_index += 1
            response = self.gateway.invoke(
                capability["capability_token"],
                CapabilityInvocation(
                    **capability["context"],
                    tool_name=name,
                    arguments=arguments,
                    call_id=f"cleanup-{call_index}",
                ),
            )
            assert response.get("ok", True), response
            return response

        call("inspect_source", {})
        structure = call("get_pdf_document_structure", {"navigation_node_id": "root"})
        for key in ("cover", "time", "shortcut"):
            call("read_evidence_node", {"node_key": key, "include_descendants": True})
        result = call("submit_evidence_revision", {
            "base_plan_artifact_id": structure["base_plan_artifact_id"],
            "operations": self.operations,
            "summary": "Remove duplicate topics, cover chapters, and false shortcut headings.",
        })
        assert result["accepted"], result
        return AgentTurnResult(
            session_id=invocation.session_id,
            final_response="Cleanup accepted.",
            finish_reason="completed",
            event_count=call_index,
            usage={},
        )

    async def cancel(self, session_id):
        return True

    async def close_session(self, session_id):
        return None


def cleanup_source(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "output")
    store = RunStore(settings.data_dir, "run_pdf_cleanup")
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    store.initialize({"target_profile": target.model_dump(mode="json"), "inputs": []})
    raw = store.put_artifact(b"PDF fixture: already normalized", kind="raw_source",
                             media_type="application/pdf", suffix=".pdf")
    manifest = SourceManifest(source_id="SRC1", canonical_input_id="IN1", display_name="Guide",
                              primary_modality="pdf", media_type="application/pdf",
                              raw_artifact_id=raw.artifact_id)
    workspace = SourceWorkspace(store, settings, AcquisitionService(store, settings), target, manifest)
    entries = [
        ("cover", "Version 1.0", "document", "Version 1.0"),
        ("date", "July 2008", "cover", "July 2008"),
        ("time", "[L]", "document", "Adjusting Time"),
        ("time_body", "Adjusting Time", "time", "Adjusting Time\n\nSet the simulation date."),
        ("shortcut", "]", "time_body", "]\n\nPress L to accelerate time. Preserve this warning."),
        ("saving", "Saving Results", "shortcut", "Saving Results\n\nSave the current simulation."),
    ]
    nodes = [EvidencePlanNode(node_key="document", title="Guide", node_type="document")]
    for ordinal, (key, title, parent, text) in enumerate(entries, 1):
        part = SourcePart(part_id=f"SRC1_section_{ordinal:04d}", source_id="SRC1",
                          kind="pdf_section", label=title, ordinal=ordinal,
                          locator={"page_start": 1 if ordinal < 3 else 3, "page_end": 3},
                          metadata={"semantic_unit": True, "evidence_default": True})
        workspace.add_part(part, text)
        nodes.append(EvidencePlanNode(
            node_key=key, title=title, node_type="section", parent_node_key=parent,
            source_refs=[EvidenceSourceRef(source_part_id=part.part_id)],
        ))
    workspace.inventory_completed = True
    engine = EvidencePlanEngine(workspace)
    plan = EvidencePlan(source_id="SRC1", nodes=nodes)
    artifacts = engine.materialize(plan)
    workspace.evidence_skeleton_plan = plan
    workspace.evidence_skeleton_plan_artifact_id = artifacts.plan_artifact_id
    workspace.evidence_skeleton_tree_artifact_id = artifacts.tree_artifact_id
    workspace.evidence_tree = artifacts.tree
    workspace.evidence_tree_artifact_id = artifacts.tree_artifact_id
    agent_view = {
        "navigation_tree": {"selected_source": "printed_toc", "nodes": []},
        "content_assignments": [
            {"evidence_node_key": node.node_key, "parent_evidence_node_key": node.parent_node_key,
             "navigation_node_id": f"layout_{node.node_key}", "title": node.title,
             "page_start": 1, "page_end": 3, "char_count": 20}
            for node in nodes[1:]
        ],
        "unassigned_content": [], "deterministically_excluded_content": [],
    }
    structure = {"document_title": "Guide", "page_count": 3, "agent_view": agent_view}
    workspace.pdf_document_structure = structure
    workspace.pdf_document_structure_artifact_id = store.put_json_artifact(
        structure, kind="pdf_document_structure").artifact_id
    normalization = store.put_json_artifact({}, kind="source_normalization_record")
    prepared = PreparedSource(workspace=workspace,
                              executor=ToolExecutor(workspace, "pdf_source_review"),
                              normalization_artifact_id=normalization.artifact_id)
    content_keys = [engine.content_key(workspace.parts[n.source_refs[0].source_part_id])
                    for n in nodes[1:3]]
    operations = [
        {"operation": "assign_content", "node_key": "document", "content_keys": content_keys},
        {"operation": "move_node", "node_key": "saving", "parent_node_key": "document"},
        {"operation": "move_node", "node_key": "time_body", "parent_node_key": "document"},
        {"operation": "merge_nodes", "node_key": "time", "source_node_keys": ["time_body"]},
        {"operation": "move_node", "node_key": "shortcut", "parent_node_key": "document"},
        {"operation": "merge_nodes", "node_key": "time", "source_node_keys": ["shortcut"]},
        {"operation": "update_node", "node_key": "time", "title": "Adjusting Time"},
    ]
    return store, prepared, operations


@pytest.mark.asyncio
async def test_pdf_review_cleans_topics_without_losing_body_or_source_ranges(tmp_path):
    store, prepared, operations = cleanup_source(tmp_path)
    workspace = prepared.workspace
    before_parts = {key: part.model_dump(mode="json") for key, part in workspace.parts.items()}
    before_text = dict(workspace.part_text)
    before_refs = Counter((ref.source_part_id, ref.start_offset, ref.end_offset)
                          for node in workspace.evidence_tree.nodes for ref in node.source_refs)
    gateway = CapabilityGateway()
    server = CapabilityGatewayServer(gateway)
    runtime = CleanupRuntime(gateway, operations)
    service = SourceReviewService(
        store=store, runtime=runtime, gateway=gateway, gateway_server=server,
        model_config=ModelRuntimeConfig(base_url="https://example.invalid/v1", model="fixture",
                                        api_key="offline-test-no-network"),
    )
    try:
        result = await service.review(prepared)
    finally:
        server.close()

    assert result.status == "succeeded"
    nodes = {node.node_key: node for node in workspace.evidence_tree.nodes}
    assert set(nodes) == {"document", "time", "saving"}
    assert "Version 1.0" in nodes["document"].content
    assert "July 2008" in nodes["document"].content
    assert "Preserve this warning." in nodes["time"].content
    assert nodes["time"].title == "Adjusting Time"
    assert "Save the current simulation." in nodes["saving"].content
    assert nodes["saving"].parent_node_id == nodes["document"].evidence_node_id
    after_refs = Counter((ref.source_part_id, ref.start_offset, ref.end_offset)
                         for node in nodes.values() for ref in node.source_refs)
    assert after_refs == before_refs
    assert before_parts == {key: part.model_dump(mode="json") for key, part in workspace.parts.items()}
    assert workspace.part_text == before_text

    events, _ = store.read_events(limit=1000)
    budget = next(event["payload"]["max_tool_calls"] for event in events
                  if event["event_type"] == "capability_tool_budget_configured")
    assert budget == PDF_SOURCE_REVIEW_MAX_TOOL_CALLS == 80
    attempt = store.list_attempts()[0]
    capsule = store.read_json_artifact(attempt.capsule_artifact_id)
    assert capsule["prompt_version"] == PDF_SOURCE_REVIEW_PROMPT_VERSION
    prompt = runtime.invocation.prompt
    for cleanup_rule in ("same-title parent and child", "version labels", "punctuation fragment",
                         "merge_nodes accepts only siblings", "must not appear in source_node_keys",
                         "merge_nodes does not rename its retained target", "use update_node",
                         "unmatched_limit=24", "actual", "next_offset", "same model turn"):
        assert cleanup_rule in prompt
    assert "bookmarks outrank" not in prompt
