from __future__ import annotations
import asyncio
import hashlib
import json
from pathlib import Path
import pytest
from bs4 import BeautifulSoup
from app.models import EvidenceExcerpt, ModelRuntimeConfig, ReferenceCandidate, TargetProfile
from app.reference_layout import (LayoutError, apply_layout, fingerprint, format_references,
                                  layout_prompt, prepare_units, presentation_errors)
from app.runtime import AgentTurnResult
from app.storage import RunStore

HTML = """<main><h1>Operators</h1><p>Use the documented operators.</p>
<table><caption>Operator rules</caption><tr><th>Operator</th><th>Alias</th><th>Precedence</th><th>Description</th></tr>
<tr><td>**</td><td>^</td><td>2</td><td>power, like pwr</td></tr>
<tr><td>||</td><td></td><td>7</td><td>boolean or</td></tr></table>
<pre>if x:
    value = "**|^"
</pre><p>See <a class="reference external" href="https://example.test">manual<svg><path d="M0 0"/></svg></a>.</p></main>"""

def setup(tmp_path: Path, html: str | None=HTML, content: str | None=None):
    store = RunStore(tmp_path, "run_layout")
    store.initialize({"target_profile":TargetProfile(software_name="Calc",target_version="1",platform="General").model_dump(mode="json"),
                      "model":ModelRuntimeConfig().public_dict(),"inputs":[]})
    locator = {}
    if html is not None:
        snapshot = store.put_json_artifact({"format":"html","content":html},
                                          kind="html_source_snapshot",preserve_text_integrity=True)
        locator["source_snapshot_artifact_id"] = snapshot.artifact_id
    if content is None:
        content = BeautifulSoup(html or "", "html.parser").get_text("")
    text_artifact = store.put_artifact(content.encode("utf-8"), kind="normalized_source_text",
                                       media_type="text/plain", suffix=".txt")
    ev = EvidenceExcerpt(evidence_id="E1",source_id="SRC1",source_part_id="part1",
        text_artifact_id=text_artifact.artifact_id,
        start_offset=0,end_offset=len(content),content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        kind="html_basic_text",title="Operators",content=content,locator=locator,extraction_method="test")
    ref = ReferenceCandidate(reference_id="R1",corpus_snapshot_id="corpus1",
        title="Operators reference",description="Check operator names and precedence.",
        reference_mode="source_catalog",source_catalog_id="C1",
        citations=[{"evidence_id":"E1","quote":content[:100]}],
        entries=[{"entry_id":"RE1","title":"source","content":content[:40_000],
                  "source_evidence_id":"E1","citations":[{"evidence_id":"E1","quote":content[:100]}]}])
    return store,ref,{"E1":ev}

def plan(units):
    return {"sections":[{"heading":"","unit_ids":[u["unit_id"] for u in units]}]}

def test_saved_html_recovers_indivisible_table_and_exact_code(tmp_path):
    store,ref,evidence = setup(tmp_path)
    original = ref.model_dump_json()
    units = prepare_units(ref,evidence,store)
    tables = [u for u in units if u["kind"]=="table"]
    assert len(tables)==1
    assert "| ** | ^ | 2 | power, like pwr |" in tables[0]["markdown"]
    assert r"| \|\| |  | 7 | boolean or |" in tables[0]["markdown"]
    assert tables[0]["locator"]["column_count"]==4
    code = next(u for u in units if u["kind"]=="code")
    assert 'if x:\n    value = "**|^"\n' in code["markdown"]
    result = apply_layout(ref,units,plan(units))
    assert result.entries==ref.entries and result.citations==ref.citations
    assert fingerprint(result)==fingerprint(ref)
    assert not presentation_errors(result)
    assert ref.model_dump_json()==original
    manifest = store.read_json_artifact(result.presentation.source_artifact_id)
    assert manifest["source_sha256"]==fingerprint(ref)
    assert manifest["units"][0]["evidence_ids"]==["E1"]

@pytest.mark.parametrize("html,reason", [
    ("<p>before</p><table><tr><th>A</th><th>B</th></tr><tr><td colspan='2'>value</td></tr></table>","spanning_table_requires_source_evidence"),
    ("<p>before</p><svg><text>diagram meaning</text></svg>","unsupported_graphic_requires_source_evidence"),
    ("<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>","table_grid_not_rectangular"),
])
def test_unsupported_structure_falls_back_to_source_without_guessing_columns(tmp_path,html,reason):
    store,ref,evidence = setup(tmp_path,html)
    units = prepare_units(ref,evidence,store)
    assert len(units)==1 and units[0]["kind"]=="source"
    assert units[0]["markdown"]==evidence["E1"].content
    assert units[0]["locator"]["recovery_reason"]==reason

def test_partial_cited_scope_cannot_pull_unrelated_html_text(tmp_path):
    store,ref,evidence=setup(tmp_path,content="Only this cited sentence.")
    units=prepare_units(ref,evidence,store)
    assert units[0]["markdown"]=="Only this cited sentence."
    assert all(u["kind"]!="table" for u in units)

def test_unheaded_table_keeps_all_data_rows_and_never_invents_column_names(tmp_path):
    store,ref,evidence=setup(tmp_path,"<table><tr><td>+</td><td>Add</td></tr><tr><td>-</td><td>Subtract</td></tr></table>")
    units=prepare_units(ref,evidence,store)
    assert units[0]["kind"]=="table"
    assert units[0]["markdown"].splitlines()==["|  |  |","| --- | --- |","| + | Add |","| - | Subtract |"]
    assert units[0]["locator"]["row_count"]==2
    assert units[0]["locator"]["header_present"] is False

def test_nested_list_is_local_structure_and_does_not_destroy_nearby_tables(tmp_path):
    html="<ul><li>Parent<ul><li>Child</li></ul></li></ul><table><tr><td>+</td><td>Add</td></tr></table>"
    store,ref,evidence=setup(tmp_path,html)
    units=prepare_units(ref,evidence,store)
    assert [unit["kind"] for unit in units]==["list","table"]
    assert units[0]["markdown"]=="- Parent\n  - Child"

def test_paragraph_and_table_breaks_are_explicit_without_spacing_adjacent_spans(tmp_path):
    html=("<p>alpha<br>beta<span>gamma</span>delta</p>"
          "<table><tr><th>Term</th><th>Meaning</th></tr><tr><td>one<br>two</td><td>literal ¶</td></tr></table>"
          "<pre>if value:\n    keep = 1\n</pre>")
    store,ref,evidence=setup(tmp_path,html)
    units=prepare_units(ref,evidence,store)
    assert units[0]["markdown"]=="alpha\nbetagammadelta"
    assert "| one<br>two | literal ¶ |" in units[1]["markdown"]
    assert 'if value:\n    keep = 1\n' in units[2]["markdown"]

@pytest.mark.parametrize("html,reason", [
    ("<p>x<sup>2</sup> is squared.</p>", "superscript_or_subscript_requires_source_evidence"),
    ("<p>x<sub>2</sub> identifies an indexed value.</p>", "superscript_or_subscript_requires_source_evidence"),
    ("<ol reversed><li>Second</li><li>First</li></ol>", "reversed_list_requires_source_evidence"),
])
def test_unproven_semantic_typography_or_numbering_keeps_original_evidence_with_warning(tmp_path,html,reason):
    store,ref,evidence=setup(tmp_path,html)
    units=prepare_units(ref,evidence,store)
    assert len(units)==1 and units[0]["kind"]=="source"
    assert units[0]["markdown"]==evidence["E1"].content
    assert units[0]["locator"]["recovery_reason"]==reason
    events=[json.loads(line) for line in store.audit_path.read_text().splitlines()]
    warning=next(e for e in events if e["event_type"]=="reference_formatting_source_fallback")
    assert warning["level"]=="warning"
    assert warning["payload"]["reason"]==reason

@pytest.mark.parametrize("change",["omit","duplicate","reorder","invent_heading","extra_fact"])
def test_plan_rejects_loss_duplication_reordering_and_free_text(tmp_path,change):
    store,ref,evidence=setup(tmp_path)
    units=prepare_units(ref,evidence,store)
    value=plan(units)
    ids=value["sections"][0]["unit_ids"]
    if change=="omit":ids.pop()
    if change=="duplicate":ids.append(ids[-1])
    if change=="reorder":ids.reverse()
    if change=="invent_heading":value["sections"][0]["heading"]="Invented condition"
    if change=="extra_fact":value["sections"][0]["markdown"]="New knowledge"
    with pytest.raises(LayoutError):apply_layout(ref,units,value)

def test_presentation_hashes_detect_edits_but_allow_validation_status_change(tmp_path):
    store,ref,evidence=setup(tmp_path)
    units=prepare_units(ref,evidence,store)
    formatted=apply_layout(ref,units,plan(units))
    assert not presentation_errors(formatted.model_copy(update={"status":"validated"}))
    assert presentation_errors(formatted.model_copy(update={"description":"A changed source description"}))
    edited=formatted.model_copy(update={"presentation":formatted.presentation.model_copy(update={"content":"Changed"})})
    assert presentation_errors(edited)

def test_generated_headings_do_not_repeat_source_heading_or_active_section(tmp_path):
    store,ref,evidence=setup(tmp_path)
    units=prepare_units(ref,evidence,store)
    value={"sections":[{"heading":"Operators","unit_ids":[units[0]["unit_id"]]},
                       {"heading":"Operators","unit_ids":[u["unit_id"] for u in units[1:]]}]}
    result=apply_layout(ref,units,value)
    assert result.presentation.content.count("## Operators")==1
    for unit in units:
        assert unit["markdown"] in result.presentation.content

def test_same_heading_in_distinct_original_units_is_still_preserved(tmp_path):
    store,ref,evidence=setup(tmp_path,"<h2>Operators</h2><p>One</p><h2>Operators</h2><p>Two</p>")
    units=prepare_units(ref,evidence,store)
    result=apply_layout(ref,units,plan(units))
    assert result.presentation.content.count("## Operators")==2

def test_section_cannot_borrow_a_heading_from_an_unrelated_source_section(tmp_path):
    store,ref,evidence=setup(tmp_path,"<h2>Operators</h2><p>Addition</p><h2>Functions</h2><p>sqrt(x)</p>")
    units=prepare_units(ref,evidence,store)
    invalid={"sections":[{"heading":"Functions","unit_ids":[u["unit_id"] for u in units]}]}
    with pytest.raises(LayoutError,match="first source unit"):
        apply_layout(ref,units,invalid)
    valid={"sections":[{"heading":"Operators","unit_ids":[u["unit_id"] for u in units[:2]]},
                       {"heading":"Functions","unit_ids":[u["unit_id"] for u in units[2:]]}]}
    result=apply_layout(ref,units,valid)
    assert result.presentation.content.count("## Operators")==1
    assert result.presentation.content.count("## Functions")==1

class Driver:
    def __init__(self, failures=0, hang=False):
        self.calls=[];self.closed=[];self.cancelled=[];self.failures=failures;self.hang=hang
    async def run(self,invocation):
        self.calls.append(invocation)
        if self.hang:await asyncio.sleep(30)
        prompt=str(invocation.prompt)
        data=json.JSONDecoder().raw_decode(prompt[prompt.index('{"reference_title":'):])[0]
        ids=[u["unit_id"] for u in data["units"]]
        if len(self.calls)<=self.failures:ids=ids[:-1]
        return AgentTurnResult(session_id=invocation.session_id,
            final_response=json.dumps({"sections":[{"heading":"","unit_ids":ids}]}),
            finish_reason="stop",usage={"input_tokens":10,"output_tokens":5})
    async def cancel(self,session):self.cancelled.append(session);return True
    async def close_session(self,session):self.closed.append(session)

@pytest.mark.asyncio
async def test_mode_switch_and_successful_retry_have_auditable_sources(tmp_path):
    store,ref,evidence=setup(tmp_path)
    runtime=Driver(failures=1)
    old=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="source_preserved")
    assert old[0] is ref and not runtime.calls
    new=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="agent_formatted")
    assert new[0].presentation and not presentation_errors(new[0])
    assert len(runtime.calls)==2 and "HOST VALIDATION ERROR" in runtime.calls[1].prompt
    assert runtime.closed==[runtime.calls[0].session_id]
    assert store.list_attempts()[0].status=="succeeded"
    events=[json.loads(l) for l in store.audit_path.read_text().splitlines()]
    assert any(e["event_type"]=="reference_formatting_completed" for e in events)

@pytest.mark.asyncio
async def test_repeated_invalid_plan_keeps_original_and_records_warning(tmp_path):
    store,ref,evidence=setup(tmp_path)
    runtime=Driver(failures=2)
    result=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="agent_formatted")
    assert result[0] is ref and len(runtime.calls)==2
    assert store.list_attempts()[0].status=="failed"
    assert "reference_formatting_fallback" in store.audit_path.read_text()

@pytest.mark.asyncio
async def test_cancel_propagates_and_closes_child_session(tmp_path):
    store,ref,evidence=setup(tmp_path)
    runtime=Driver(hang=True)
    task=asyncio.create_task(format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="agent_formatted"))
    while not runtime.calls:await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert store.list_attempts()[0].status=="cancelled"
    assert runtime.cancelled and runtime.closed

@pytest.mark.asyncio
async def test_timeout_keeps_original_and_cancels_runtime(tmp_path):
    store,ref,evidence=setup(tmp_path)
    runtime=Driver(hang=True)
    result=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="agent_formatted",timeout_seconds=.01)
    assert result[0] is ref
    assert runtime.cancelled and runtime.closed
    assert store.list_attempts()[0].status=="failed"

@pytest.mark.asyncio
async def test_large_context_never_calls_model(tmp_path):
    store,ref,evidence=setup(tmp_path,None,content="x"*160_001)
    runtime=Driver()
    result=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=runtime,model=ModelRuntimeConfig(),mode="agent_formatted")
    assert result[0] is ref and not runtime.calls

@pytest.mark.asyncio
async def test_runtime_error_does_not_persist_credentials(tmp_path):
    store,ref,evidence=setup(tmp_path)
    class BadDriver(Driver):
        async def run(self,invocation):raise RuntimeError("Secret Bearer test-credential-value")
    result=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=BadDriver(),model=ModelRuntimeConfig(),mode="agent_formatted")
    assert result[0] is ref
    assert "test-credential-value" not in store.audit_path.read_text()

@pytest.mark.asyncio
async def test_optional_session_cleanup_failure_keeps_successful_presentation(tmp_path):
    store,ref,evidence=setup(tmp_path)
    class CloseErrorDriver(Driver):
        async def close_session(self,session):raise RuntimeError("Sensitive cleanup detail")
    result=await format_references(references=[ref],evidence_by_id=evidence,store=store,
        runtime=CloseErrorDriver(),model=ModelRuntimeConfig(),mode="agent_formatted")
    assert result[0].presentation and not presentation_errors(result[0])
    assert "reference_formatting_session_close_failed" in store.audit_path.read_text()
    assert "Sensitive cleanup detail" not in store.audit_path.read_text()

