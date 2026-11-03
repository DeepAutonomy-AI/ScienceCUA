"""Bounded structural facts and a one-submit guided Evidence review."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.capabilities import CapabilityGateway, CapabilityInvocation
from app.html_guided.review import BoundGuidedReviewExecutor, GUIDED_REVIEW_PROMPT, guided_review_context
from app.models import TargetProfile
from app.orchestration.html_guided_stage import normalize_guided_html_source
from app.orchestration.source_stage import SourceReviewService
from app.runtime import AgentTurnResult, FakeAgentRuntime
from test_html_guided_stage import PlanningRuntime, stage_inputs


def test_large_review_overview_keeps_existing_keys_and_authored_groups_within_budget():
    target = TargetProfile(software_name="Example", target_version="1", platform="General")
    headings = [{"level": 2, "label": "Category A"}, {"level": 2, "label": "Category B"}]
    headings += [{"level": 3, "label": f"Topic {index}: " + "Authored detail " * 24} for index in range(600)]
    facts = {
        "document_title": "Example Manual", "headings": headings,
        "disclosures": [{"summary": "Category A"}, {"summary": "Category B"}],
        "nested_link_trees": [{"label": f"Entry {index}: " + "linked operation " * 25,
                               "href": f"https://example.org/articles/{index}.html", "in_scope": False,
                               "children": []} for index in range(500)],
    }
    fields = {"node_key": "document", "parent_node_key": None, "title": "Example Manual", "node_type": "document", "child_count": 600, "direct_char_count": 40}
    nodes = [{**fields, "node_key": f"existing_{index:03d}", "parent_node_key": "document",
              "title": f"Source chapter {index}: " + "操作说明 " * 25, "node_type": "section", "child_count": 0}
             for index in range(200)]
    tree_nodes = [SimpleNamespace(evidence_node_id="N0", parent_node_id=None, node_key="document",
                                  title="Example Manual", node_type="document", direct_char_count=40)]
    tree_nodes += [SimpleNamespace(evidence_node_id=f"N{index + 1}",
                                  parent_node_id="N0" if index < 2 else "N1",
                                  node_key=item["node_key"], title=item["title"],
                                  node_type="page" if index < 2 else "section", direct_char_count=40)
                   for index, item in enumerate(nodes)]

    def skeleton(**kwargs):
        assert kwargs == {"depth": 3, "limit": 200}
        # A depth-first inventory puts the second primary page after the first page's many sections.
        return {"base_plan_artifact_id": "art_existing_plan", "selected_node": fields, "nodes": nodes[:1] + nodes[2:] + nodes[1:2],
                "truncated": True, "total_descendants_in_view": 600}

    executor = SimpleNamespace(
        _get_root_structure=lambda: {"root_structure": facts}, _get_evidence_skeleton=skeleton,
        workspace=SimpleNamespace(
            target_profile=target, gaps=["One documented chapter was not fetched"],
            evidence_tree=SimpleNamespace(root_node_id="N0", nodes=tree_nodes),
            manifest=SimpleNamespace(source_id="SRC1", source_metadata={"acquisition_coverage": {
                "page_count": 40, "child_page_count": 39, "acquisition_complete": False, "truncated": True,
            }}),
        ),
    )
    context = guided_review_context(executor)
    serialized = json.dumps(context, ensure_ascii=False)
    assert len(serialized) <= 50_000
    assert len(serialized.encode("utf-8")) <= context["context_budget_bytes"] <= 50_000
    assert "in_scope" not in serialized
    assert "base_plan_artifact_id" not in context
    assert "Python binds" in context["plan_binding"]
    assert context["overview_truncated"] is True
    assert context["nodes"], "The bounded reviewer must retain actionable candidate node keys"
    assert context["nodes"][0]["node_key"] == "existing_000"
    assert context["nodes"][1]["node_key"] == "existing_001"
    assert {item["node_key"] for item in context["nodes"]} <= {item["node_key"] for item in nodes}
    assert {item["summary"] for item in context["root_structure"]["disclosures"]} >= {"Category A", "Category B"}
    assert context["acquisition"]["acquisition_complete"] is False
    assert context["reported_gap_count"] == 1


class SubmitOnlyReviewRuntime(FakeAgentRuntime):
    def __init__(self, gateway):
        super().__init__()
        self.gateway = gateway
        self.calls = []
        self.context = None

    async def run(self, invocation):
        self.invocations.append(invocation)
        assert invocation.role == "source_review"
        assert set(invocation.allowed_tools) == {"read_evidence_node", "submit_evidence_revision"}
        assert invocation.model["reasoning_effort"] == "low"
        assert str(invocation.prompt).startswith(GUIDED_REVIEW_PROMPT)
        self.context, _ = json.JSONDecoder().raw_decode(str(invocation.prompt)[len(GUIDED_REVIEW_PROMPT):])
        capability = invocation.metadata["capability"]
        self.calls.append("submit_evidence_revision")
        response = self.gateway.invoke(capability["capability_token"], CapabilityInvocation(
            **capability["context"], tool_name="submit_evidence_revision", call_id="review.submit.1",
            arguments={"operations": [],
                       "summary": "The supplied authored chapter hierarchy is coherent."},
        ))
        assert response["accepted"] is True
        return AgentTurnResult(session_id=invocation.session_id, final_response="Hierarchy accepted.", usage={"inputTokens": 20, "outputTokens": 4})


@pytest.mark.asyncio
async def test_guided_review_commits_empty_operations_without_identity_or_root_tool_reads(tmp_path, monkeypatch):
    store, settings, acquisition, target, manifest, model = stage_inputs(tmp_path, monkeypatch)
    prepared = await normalize_guided_html_source(
        store, settings, acquisition, target, manifest, runtime=PlanningRuntime(), model_config=model,
    )
    before_parts = {key: part.model_dump(mode="json") for key, part in prepared.workspace.parts.items()}
    before_text = dict(prepared.workspace.part_text)
    original_tool_surface = set(prepared.executor.source_review_tools)
    assert original_tool_surface == {"inspect_source", "get_root_structure", "read_evidence_node", "submit_evidence_revision"}
    bound = BoundGuidedReviewExecutor(prepared.executor)
    submit_spec = next(spec['function'] for spec in bound.tool_specs()
                       if spec['function']['name'] == 'submit_evidence_revision')
    assert 'base_plan_artifact_id' not in submit_spec['parameters']['properties']
    assert 'base_plan_artifact_id' not in submit_spec['parameters']['required']
    original_submit = next(spec['function'] for spec in prepared.executor.tool_specs()
                           if spec['function']['name'] == 'submit_evidence_revision')
    assert 'base_plan_artifact_id' in original_submit['parameters']['required']
    assert bound.call('submit_evidence_revision', {'base_plan_artifact_id': 'art_not_the_bound_plan',
                                                  'operations': []})['error'] == 'bound_plan_override_denied'
    assert prepared.workspace.final_result is None
    plan_id = prepared.workspace.evidence_skeleton_plan_artifact_id
    prepared.workspace.evidence_skeleton_plan_artifact_id = 'another_plan'
    assert bound.call('submit_evidence_revision', {'operations': []})['error'] == 'bound_plan_changed'
    prepared.workspace.evidence_skeleton_plan_artifact_id = plan_id
    gateway = CapabilityGateway()
    runtime = SubmitOnlyReviewRuntime(gateway)
    # The fake runtime invokes the real authorization gateway directly; no socket or model API is needed.
    service = SourceReviewService(
        store=store, runtime=runtime, gateway=gateway,
        gateway_server=SimpleNamespace(endpoint_url="http://127.0.0.1:9/invoke"), model_config=model,
    )
    result = await service.review(prepared)
    assert result.status == "succeeded"
    assert runtime.calls == ["submit_evidence_revision"]
    assert len(runtime.invocations) == 1
    assert "base_plan_artifact_id" not in runtime.context
    assert "Python binds" in runtime.context["plan_binding"]
    assert runtime.context["nodes"]
    assert {key: part.model_dump(mode="json") for key, part in prepared.workspace.parts.items()} == before_parts
    assert prepared.workspace.part_text == before_text
    assert set(prepared.executor.source_review_tools) == original_tool_surface
    review_attempts = [attempt for attempt in store.list_attempts() if attempt.stage == "source_review"]
    assert len(review_attempts) == 1 and review_attempts[0].status == "succeeded"
    assert set(store.read_session(review_attempts[0].child_session_id).allowed_tools) == {"read_evidence_node", "submit_evidence_revision"}
    assert any(node.title == "Open data" for node in result.evidence_tree.nodes)
    assert any("Choose File, then Save." in node.content for node in result.evidence_tree.nodes)
    bound_events = [json.loads(line) for line in store.audit_path.read_text('utf-8').splitlines()
                    if json.loads(line)['event_type'] == 'html_guided_review_plan_bound']
    assert bound_events and all(event['payload']['base_plan_artifact_id'] == plan_id for event in bound_events)
