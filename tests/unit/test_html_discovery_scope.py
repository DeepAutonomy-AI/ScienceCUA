"""Discovery gates prove observed scope facts without choosing scope for the model."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.orchestration.source_discovery_stage import SourceDiscoveryToolExecutor
from app.source_discovery_contracts import SourceCandidate, canonical_candidate_url
from app.storage import RunStore


ENTRY = "https://science.example.org/guide/"
OTHER = "https://science.example.org/guide/overview.html"


def scope(**changes):
    return {
        "kind": "tutorial_collection",
        "scope_summary": "The entry collects the author's image-analysis lessons.",
        "entry_evidence_ids": ["H1", "L1"],
        "included_sections": ["Loading images", "Segmentation"],
        "excluded_sections": ["Developer API reference"],
        "comparisons": [],
        "comparison_note": "No distinct adjacent entry was established by the available links.",
        "limitations": [],
        **changes,
    }


def candidate(**changes):
    return {
        "title": "Scientific application lessons",
        "url": ENTRY,
        "modality": "html",
        "source_type": "official",
        "document_version": "unknown",
        "version_relation": "unknown",
        "version_evidence": "The entry does not declare a software release.",
        "root_assessment": "The entry's main body links to the included lessons.",
        "reason": "The model selected this collection for the requested learning goal.",
        "html_scope": scope(),
        **changes,
    }


class ObservedProbe:
    """Only the host probe can create these capture/snapshot references."""

    def __init__(self, store):
        self.store = store
        self.results = {}
        self.calls = []

    def html(self, url=ENTRY, *, final_url=None, evidence_ids=("H1", "L1", "P1")):
        final_url = final_url or url
        heading, link, main = evidence_ids
        snapshot = {
            "url": final_url,
            "title": "Scientific application lessons",
            "headings": [{"evidence_id": heading, "text": "Image analysis lessons", "level": 1}],
            "links": [{"evidence_id": link, "url": OTHER, "text": "Loading images"}],
            "main_content": {"evidence_id": main, "text": "Choose a lesson to begin."},
        }
        raw = self.store.put_artifact(
            b"<main><h1>Image analysis lessons</h1><a href='overview.html'>Loading images</a></main>",
            kind="discovery_html_capture", media_type="text/html", suffix=".html",
            metadata={"url": url, "final_url": final_url, "complete": True},
        )
        artifact = self.store.put_json_artifact(snapshot, kind="discovery_html_snapshot")
        result = {
            "status": "available", "url": url, "final_url": final_url,
            "modality": "html", "http_status": 200, "complete": True,
            "bytes_checked": 98, "title": snapshot["title"], "preview": "Image analysis lessons",
            "html_snapshot": snapshot, "raw_html_artifact_id": raw.artifact_id,
            "html_snapshot_artifact_id": artifact.artifact_id,
        }
        self.results[url] = result
        return result

    def inspect(self, url):
        self.calls.append(url)
        return deepcopy(self.results[url])


@pytest.fixture
def environment(tmp_path):
    store = RunStore(tmp_path, "run_html_discovery_scope")
    store.initialize({"target_profile": {
        "software_name": "Scientific application", "target_version": "1.0", "platform": "General",
    }, "inputs": [], "model": {}})
    probe = ObservedProbe(store)
    probe.html()
    return store, probe, SourceDiscoveryToolExecutor(store, "attempt_scope", "English", probe=probe)


def nominate(executor, tool, item):
    args = {"candidate": item} if tool == "save_source_candidate" else {
        "candidates": [item], "summary": "Selected the model-assessed teaching collection."
    }
    return executor.call(tool, args)


@pytest.mark.parametrize("tool", ["save_source_candidate", "submit_source_candidates"])
def test_new_html_save_and_submit_require_host_inspection(environment, tool):
    _, _, executor = environment
    executor.observe_search_sources([{"url": ENTRY, "title": "Lessons", "snippet": "Full course"}])
    response = nominate(executor, tool, candidate())
    assert response["error"] == "html_candidate_inspection_required"
    assert not executor.has_saved_candidates and not executor.is_finished


@pytest.mark.parametrize("tool", ["save_source_candidate", "submit_source_candidates"])
def test_new_html_save_and_submit_require_model_scope_after_inspection(environment, tool):
    _, _, executor = environment
    assert executor.call("inspect_source_candidate", {"url": ENTRY})["status"] == "available"
    response = nominate(executor, tool, candidate(html_scope=None))
    assert response["error"] == "html_scope_assessment_required"
    assert not executor.has_saved_candidates and not executor.is_finished


@pytest.mark.parametrize("observed", [
    {"status": "failed", "url": ENTRY, "error": "http_error", "http_status": 403},
    {"status": "available", "url": ENTRY, "modality": "pdf", "header_verified": True},
])
def test_html_candidate_must_be_an_available_html_response(environment, observed):
    _, probe, executor = environment
    probe.results[ENTRY] = observed
    executor.call("inspect_source_candidate", {"url": ENTRY})
    response = nominate(executor, "save_source_candidate", candidate())
    assert response["error"] == "html_candidate_unavailable"
    assert not executor.has_saved_candidates


def test_scope_evidence_is_bound_to_the_entry_not_another_inspected_page(environment):
    _, probe, executor = environment
    probe.html(OTHER, evidence_ids=("H9", "L9", "P9"))
    executor.call("inspect_source_candidate", {"url": ENTRY})
    executor.call("inspect_source_candidate", {"url": OTHER})
    invalid = nominate(executor, "save_source_candidate", candidate(html_scope=scope(entry_evidence_ids=["H9"])))
    assert invalid["error"] == "html_scope_evidence_invalid"
    assert not executor.has_saved_candidates
    valid = nominate(executor, "save_source_candidate", candidate(html_scope=scope(entry_evidence_ids=["P1"])))
    assert valid["ok"]


@pytest.mark.parametrize("comparison_state", ["not_inspected", "skipped", "failed", "pdf"])
def test_comparisons_require_an_actual_available_html_inspection(environment, comparison_state):
    _, probe, executor = environment
    executor.call("inspect_source_candidate", {"url": ENTRY})
    executor.observe_search_sources([{"url": OTHER, "title": "Broader overview"}])
    if comparison_state == "skipped":
        executor.call("inspect_source_candidate", {"url": OTHER, "action": "skip", "reason": "Unrelated overview"})
    elif comparison_state in {"failed", "pdf"}:
        probe.results[OTHER] = {"status": "failed", "url": OTHER, "error": "timeout"} if comparison_state == "failed" else {
            "status": "available", "url": OTHER, "modality": "pdf", "header_verified": True,
        }
        executor.call("inspect_source_candidate", {"url": OTHER})
    compared = scope(comparisons=[{"url": OTHER, "relation": "broader", "reason": "Also includes developer references."}])
    response = nominate(executor, "save_source_candidate", candidate(html_scope=compared))
    assert response["error"] == "html_scope_comparison_unverified"
    assert not executor.has_saved_candidates


def test_verified_comparison_is_retained_as_model_assessment(environment):
    store, probe, executor = environment
    probe.html(OTHER)
    executor.call("inspect_source_candidate", {"url": ENTRY})
    executor.call("inspect_source_candidate", {"url": OTHER})
    assessment = scope(comparisons=[{"url": OTHER, "relation": "broader", "reason": "Contains API material outside the requested lessons."}])
    response = nominate(executor, "submit_source_candidates", candidate(html_scope=assessment))
    assert response["ok"]
    persisted = store.read_json_artifact(response["candidate_artifact_id"])["candidates"][0]
    assert persisted["html_scope"]["comparisons"] == assessment["comparisons"]
    assert persisted["html_scope"]["kind"] == "tutorial_collection"


def test_no_comparison_page_is_allowed_with_an_explicit_reason(environment):
    _, _, executor = environment
    executor.call("inspect_source_candidate", {"url": ENTRY})
    response = nominate(executor, "save_source_candidate", candidate())
    assert response["ok"]
    assert executor.has_saved_candidates and not executor.is_finished


def test_empty_comparison_note_does_not_satisfy_a_new_scope(environment):
    _, _, executor = environment
    executor.call("inspect_source_candidate", {"url": ENTRY})
    response = nominate(executor, "save_source_candidate", candidate(html_scope=scope(comparison_note="")))
    assert not response["ok"]
    assert not executor.has_saved_candidates


@pytest.mark.parametrize("finish", ["deadline", "empty_submit"])
def test_saved_scoped_candidate_survives_wrap_up_and_rejected_update(environment, finish):
    store, probe, executor = environment
    observed = executor.call("inspect_source_candidate", {"url": ENTRY})
    saved = nominate(executor, "save_source_candidate", candidate())
    assert saved["ok"]
    checkpoint = store.read_json_artifact(saved["candidate_artifact_id"])
    rejected = nominate(executor, "save_source_candidate", candidate(html_scope=scope(entry_evidence_ids=["invented"])))
    assert rejected["error"] == "html_scope_evidence_invalid"
    if finish == "deadline":
        result = executor.finalize_saved("time_budget_reached")
        artifact_id = result.candidate_artifact_id
        assert store.read_json_artifact(artifact_id)["completion_reason"] == "time_budget_reached"
    else:
        response = executor.call("submit_source_candidates", {"candidates": [], "summary": "Keep the verified collection."})
        assert response["ok"]
        artifact_id = response["candidate_artifact_id"]
    persisted = store.read_json_artifact(artifact_id)["candidates"]
    assert len(persisted) == 1 and persisted == checkpoint["candidates"]
    assert persisted[0]["html_scope"] == scope()
    assert persisted[0]["verification"]["html_snapshot_artifact_id"] == observed["html_snapshot_artifact_id"]
    assert persisted[0]["verification"]["raw_html_artifact_id"] == observed["raw_html_artifact_id"]
    assert probe.calls == [ENTRY]


def test_inspection_cache_avoids_a_second_fetch_for_the_same_canonical_page(environment):
    _, probe, executor = environment
    first = executor.call("inspect_source_candidate", {"url": ENTRY})
    second = executor.call("inspect_source_candidate", {"url": ENTRY + "#contents"})
    assert first["html_snapshot_artifact_id"] == second["html_snapshot_artifact_id"]
    assert probe.calls == [ENTRY]


def test_model_can_explicitly_replace_a_saved_narrow_entry_without_losing_its_checkpoint(environment):
    store, probe, executor = environment
    narrow = "https://science.example.org/guide/first-steps.html"
    probe.html(narrow)
    executor.call("inspect_source_candidate", {"url": narrow})
    first = nominate(executor, "save_source_candidate", candidate(
        url=narrow, html_scope=scope(kind="single_page_manual"),
    ))
    assert first["ok"]
    original_checkpoint = store.read_json_artifact(first["candidate_artifact_id"])
    executor.call("inspect_source_candidate", {"url": ENTRY})
    replacement = candidate(html_scope=scope(comparisons=[{
        "url": narrow, "relation": "narrower", "reason": "The earlier entry teaches only the first procedure.",
    }]))
    response = executor.call("save_source_candidate", {"candidate": replacement, "replaces_url": narrow})
    assert response["ok"] and response["candidate_count"] == 1
    updated = store.read_json_artifact(response["candidate_artifact_id"])["candidates"]
    assert [item["url"] for item in updated] == [ENTRY]
    assert store.read_json_artifact(first["candidate_artifact_id"]) == original_checkpoint
    assert original_checkpoint["candidates"][0]["url"] == narrow
    final = executor.call("submit_source_candidates", {"candidates": [], "summary": "Retain the inspected collection."})
    assert final["ok"]
    assert [item.url for item in executor.result.candidates] == [ENTRY]


def test_unverified_replacement_does_not_delete_a_valid_saved_entry(environment):
    store, _, executor = environment
    executor.call("inspect_source_candidate", {"url": ENTRY})
    first = nominate(executor, "save_source_candidate", candidate())
    assert first["ok"]
    rejected = executor.call("save_source_candidate", {"candidate": candidate(url=OTHER), "replaces_url": ENTRY})
    assert rejected["error"] == "html_candidate_inspection_required"
    final = executor.finalize_saved("time_budget_reached")
    assert [item.url for item in final.candidates] == [ENTRY]
    assert store.read_json_artifact(final.candidate_artifact_id)["candidates"] == store.read_json_artifact(
        first["candidate_artifact_id"]
    )["candidates"]


def test_only_observed_redirects_prove_entry_and_comparison_aliases(environment):
    _, probe, executor = environment
    requested = "https://science.example.org/manual"
    final = "https://science.example.org/manual/index.html"
    comparison_requested = "https://science.example.org/all-docs"
    comparison_final = "https://science.example.org/documentation/"
    probe.html(requested, final_url=final)
    probe.html(comparison_requested, final_url=comparison_final)
    executor.call("inspect_source_candidate", {"url": requested})
    executor.call("inspect_source_candidate", {"url": comparison_requested})
    assessment = scope(comparisons=[{
        "url": comparison_final, "relation": "broader", "reason": "The observed redirect leads to the broader reference collection.",
    }])
    response = nominate(executor, "save_source_candidate", candidate(url=final, html_scope=assessment))
    assert response["ok"]
    unobserved_alias = "https://science.example.org/manual/"
    assert canonical_candidate_url(unobserved_alias) != canonical_candidate_url(final)
    rejected = nominate(executor, "save_source_candidate", candidate(url=unobserved_alias))
    assert rejected["error"] == "html_candidate_inspection_required"
    query_change = nominate(executor, "save_source_candidate", candidate(url=final + "?version=other"))
    assert query_change["error"] == "html_candidate_inspection_required"


@pytest.mark.parametrize("path,kind", [
    ("/", "documentation_collection"),
    ("/getting_started.html", "single_page_manual"),
    ("/tutorials/index.html", "user_manual"),
    ("/deep/chapter-03/lesson.html", "tutorial_collection"),
    ("/download/manual.pdf", "single_page_manual"),
])
def test_host_does_not_decide_scope_from_url_depth_keywords_or_extension(environment, path, kind):
    _, probe, executor = environment
    url = "https://science.example.org" + path
    probe.html(url)
    executor.call("inspect_source_candidate", {"url": url})
    response = nominate(executor, "submit_source_candidates", candidate(url=url, html_scope=scope(kind=kind)))
    assert response["ok"]
    assert executor.result.candidates[0].html_scope.kind == kind


def test_historical_persisted_html_without_scope_remains_readable_but_cannot_be_newly_saved(environment):
    _, _, executor = environment
    old = candidate()
    old.pop("html_scope")
    persisted = SourceCandidate.model_validate({**old, "candidate_id": "candidate_0123456789abcdef", "verification": {}})
    assert persisted.html_scope is None
    executor.call("inspect_source_candidate", {"url": ENTRY})
    response = nominate(executor, "save_source_candidate", old)
    assert response["error"] == "html_scope_assessment_required"


def test_pdf_gate_does_not_require_html_scope(environment):
    _, probe, executor = environment
    url = "https://science.example.org/manual.pdf"
    probe.results[url] = {"status": "available", "url": url, "final_url": url, "modality": "pdf", "header_verified": True}
    executor.call("inspect_source_candidate", {"url": url})
    response = nominate(executor, "submit_source_candidates", candidate(url=url, modality="pdf", html_scope=None))
    assert response["ok"]


def test_nested_scope_tool_schemas_have_no_unresolved_model_references(environment):
    _, _, executor = environment

    def refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                yield value["$ref"]
            for item in value.values():
                yield from refs(item)
        elif isinstance(value, list):
            for item in value:
                yield from refs(item)

    for tool in executor.tool_specs():
        parameters = tool["function"]["parameters"]
        assert not list(refs(parameters)), tool["function"]["name"]
        if tool["function"]["name"] in {"save_source_candidate", "submit_source_candidates"}:
            assert "html_scope" in str(parameters)
            assert "entry_evidence_ids" in str(parameters)
            assert "comparison_note" in str(parameters)
            candidate_schema = (parameters["properties"]["candidate"] if tool["function"]["name"] == "save_source_candidate"
                                else parameters["properties"]["candidates"]["items"])
            html_scope = candidate_schema["properties"]["html_scope"]
            assert html_scope["type"] == "object"
            assert "anyOf" not in html_scope
            assert html_scope["properties"]["comparisons"]["items"]["type"] == "object"
            assert "html_scope" not in candidate_schema.get("required", [])


@pytest.mark.parametrize("field", ["scope_summary", "comparison_note"])
def test_whitespace_cannot_replace_a_scope_explanation(environment, field):
    _, _, executor = environment
    executor.call("inspect_source_candidate", {"url": ENTRY})
    value = scope(**{field: "   "})
    assert not nominate(executor, "save_source_candidate", candidate(html_scope=value))["ok"]
