from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from app.capabilities import CapabilityGateway
from app.models import ModelRuntimeConfig
from app.orchestration.reference_stage import ReferenceBuildService
from app.runtime import AgentTurnResult
from app.knowledge import validate_references
from test_reference_review_submission import _workspace
from test_t6_reference_stage import ReferenceDrivingRuntime

@pytest.mark.asyncio
@pytest.mark.parametrize("formatting_succeeds", [True, False])
async def test_opt_in_stage_formats_after_review_and_keeps_valid_sources(tmp_path, formatting_succeeds):
    workspace = _workspace(tmp_path, catalog_count=1)
    store = workspace.store
    model = ModelRuntimeConfig()
    store.initialize({"target_profile": workspace.target_profile.model_dump(mode="json"),
                      "model": model.public_dict(), "inputs": []})
    corpus_artifact = store.put_json_artifact(workspace.corpus.model_dump(mode="json"), kind="corpus_snapshot")
    gateway = CapabilityGateway()

    class Driver(ReferenceDrivingRuntime):
        async def run(self, invocation):
            if invocation.role == "reference_formatting":
                self.invocations.append(invocation)
                return AgentTurnResult(session_id=invocation.session_id,
                    final_response=json.dumps({"sections": [{"heading": "", "unit_ids": ["U1"]}]} if formatting_succeeds else {"sections": []}),
                    finish_reason="stop")
            return await super().run(invocation)

    driver = Driver(gateway)
    service = ReferenceBuildService(store=store,
        settings=replace(workspace.settings, reference_formatting_mode="agent_formatted"),
        runtime=driver, gateway=gateway,
        gateway_server=SimpleNamespace(endpoint_url="http://unused.invalid/invoke"),
        model_config=model)
    result = await service.build(target_profile=workspace.target_profile, corpus=workspace.corpus,
                                 corpus_artifact_id=corpus_artifact.artifact_id)
    assert len(result.references) == 1
    ref = result.references[0]
    if not formatting_succeeds:
        assert ref.presentation is None
        assert result.stage_result["status"] == "degraded"
        assert any("kept original" in warning for warning in result.stage_result["warnings"])
        assert len(driver.invocations) == 3
        return
    assert ref.presentation is not None
    assert ref.presentation.content.strip() == workspace.corpus.evidence[0].content.strip()
    assert ref.entries and ref.entries[0].content in workspace.corpus.evidence[0].content
    assert len(driver.invocations) == 2
    assert driver.invocations[-1].allowed_tools == ()
    accepted, checks = validate_references([ref], workspace.evidence_by_id)
    assert accepted and all(c.accepted for c in checks)
    saved = store.read_json_artifact(result.validated_artifact_id)
    assert saved["references"][0]["presentation"]["mode"] == "agent_formatted"
    original = store.read_json_artifact(result.candidate_artifact_id)
    assert original["references"][0]["presentation"] is None

@pytest.mark.asyncio
async def test_invalid_mode_fails_before_model_or_attempt(tmp_path):
    workspace = _workspace(tmp_path)
    gateway = CapabilityGateway()
    driver = ReferenceDrivingRuntime(gateway)
    service = ReferenceBuildService(store=workspace.store,
        settings=replace(workspace.settings, reference_formatting_mode="typo"),
        runtime=driver, gateway=gateway,
        gateway_server=SimpleNamespace(endpoint_url="http://unused.invalid/invoke"),
        model_config=ModelRuntimeConfig())
    with pytest.raises(ValueError, match="reference_formatting_mode"):
        await service.build(target_profile=workspace.target_profile, corpus=workspace.corpus,
                            corpus_artifact_id="unused")
    assert not driver.invocations

