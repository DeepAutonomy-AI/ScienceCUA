"""Authored HTML directories affect Evidence topology, never frozen SourceParts."""
from collections import Counter
from copy import deepcopy

import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.evidence_planning import EvidencePlanEngine
from app.models import StagedInput, TargetProfile
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


BASE = "https://docs.example.test/book/"


def _directory(style):
    if style == "dl":
        return '''<dl class="toc">
        <dt><span><strong><a href="intro.html">1. Introduction</a></strong></span></dt>
        <dt><span><a href="graphs.html">2. Graphs</a></span></dt>
        <dd><div><dl><dt><a href="graphs.html#syntax">Syntax</a></dt>
        <dt><a href="features.html">Features</a></dt></dl></div></dd>
        <dt><span><a href="commands.html">3. Commands</a></span></dt>
        <dd><dl><dt><a href="commands.html#plus">plus</a></dt>
        <dt><a href="times.html">times</a></dt><dt><a href="minus.html">minus</a></dt></dl></dd>
        </dl>'''
    return f'''<{style} class="toc">
        <li><span><strong><a href="intro.html">1. Introduction</a></strong></span></li>
        <li><a href="graphs.html">2. Graphs</a><div><{style}>
        <li><a href="graphs.html#syntax">Syntax</a></li>
        <li><a href="features.html">Features</a></li></{style}></div></li>
        <li><a href="commands.html">3. Commands</a><{style}>
        <li><a href="commands.html#plus">plus</a></li>
        <li><a href="times.html">times</a></li><li><a href="minus.html">minus</a></li>
        </{style}></li></{style}>'''


def _workspace(tmp_path, directory, page_names, extra_body=None):
    root = f'''<!-- saved from url=(0041){BASE}index.html -->
    <html><head><title>Scientific Handbook</title></head><body><main>
    <h1>Scientific Handbook</h1><p>Application overview.</p>
    <div class="toc"><p><b>Table of Contents</b></p>{directory}</div>
    </main></body></html>'''.encode()
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out", max_html_pages=0)
    store = RunStore(settings.data_dir, "run_authored_directory")
    path = store.stage_upload("I1", "handbook.html", root)
    target = TargetProfile(software_name="Scientific App", target_version="1.0", platform="General")
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id="I1", kind="uploaded_file",
        display_name="handbook.html", staged_path=str(path), declared_media_type="text/html")], target).manifests[0]
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    executor = ToolExecutor(workspace, "directory_test")
    result = executor.call("html_inventory", {"max_pages": 0})
    assert result["ok"], result
    for ordinal, name in enumerate(page_names, 2):
        body = f"<h1>{name}</h1><p>Exact instructions for {name}.</p>"
        if name == "graphs":
            body += '<h2 id="syntax">Syntax</h2><pre>plot(x + 1)\n</pre>'
        if name == "commands":
            body += '<h2 id="plus">plus</h2><p>Add two numbers.</p>'
        body += (extra_body or {}).get(name, "")
        data = f"<html><head><title>{name}</title></head><body><main>{body}</main></body></html>".encode()
        artifact = store.put_artifact(data, kind="raw_html", media_type="text/html", suffix=".html", source_id=manifest.source_id)
        executor._materialize_html_page(page_url=BASE + name + ".html", data=data,
            artifact_id=artifact.artifact_id, ordinal=ordinal, navigation_depth=1,
            capability_branch="Scientific Handbook", parent_url=BASE + "index.html")
    return workspace, result


def _children(plan, parent):
    return [node.title for node in plan.nodes if node.parent_node_key == parent]


@pytest.mark.parametrize("style", ["dl", "ul", "ol"])
def test_nested_directories_preserve_order_fragments_and_inline_wrappers(tmp_path, style):
    workspace, result = _workspace(tmp_path, _directory(style), [])
    structure = workspace.store.read_json_artifact(result["root_structure_artifact_id"])
    assert len(structure["nested_link_trees"]) == 1
    tree = structure["nested_link_trees"][0]
    assert tree["tag"] == style
    assert tree["context_label"] == "Table of Contents"
    assert [item["label"] for item in tree["items"]] == ["1. Introduction", "2. Graphs", "3. Commands"]
    assert [item["label"] for item in tree["items"][1]["children"]] == ["Syntax", "Features"]
    assert tree["items"][1]["children"][0]["href"] == BASE + "graphs.html#syntax"
    assert tree["items"][1]["children"][0]["fragment"] == "syntax"
    assert [item["label"] for item in tree["items"][2]["children"]] == ["plus", "times", "minus"]


@pytest.mark.parametrize("style", ["dl", "ul", "ol"])
def test_out_of_order_acquisition_changes_neither_authored_hierarchy_nor_frozen_parts(tmp_path, style):
    # Deliberately capture descendants before parents, just as random sampling can.
    workspace, _ = _workspace(tmp_path, _directory(style),
        ["minus", "features", "times", "commands", "graphs", "intro"])
    parts_before = deepcopy({key: part.model_dump(mode="json") for key, part in workspace.parts.items()})
    text_before = dict(workspace.part_text)
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    by_title = {node.title: node for node in plan.nodes}
    assert _children(plan, "document") == ["intro", "graphs", "commands"]
    assert _children(plan, by_title["graphs"].node_key) == ["Syntax", "features"]
    assert _children(plan, by_title["commands"].node_key) == ["plus", "times", "minus"]
    refs = [ref.source_part_id for node in plan.nodes for ref in node.source_refs]
    assert Counter(refs) == Counter(part.part_id for part in engine.eligible_parts())
    materialized = engine.materialize(plan)
    assert len(materialized.tree.nodes) == 9
    assert {key: part.model_dump(mode="json") for key, part in workspace.parts.items()} == parts_before
    assert workspace.part_text == text_before
    for part in engine.eligible_parts():
        assert workspace.store.artifact_path(part.text_artifact_id).read_text("utf8") == text_before[part.part_id]


def test_missing_parent_is_a_source_labeled_group_with_no_fabricated_evidence(tmp_path):
    workspace, _ = _workspace(tmp_path, _directory("dl"), ["minus", "times"])
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    by_title = {node.title: node for node in plan.nodes}
    parent = by_title["3. Commands"]
    assert parent.source_refs == []
    assert _children(plan, parent.node_key) == ["times", "minus"]
    assert _children(plan, "document") == ["3. Commands"]
    assert "intro" not in by_title
    assert engine.validate(plan).valid
    engine.materialize(plan)


def test_duplicate_links_and_same_page_fragments_cannot_create_cycles_or_duplicate_content(tmp_path):
    directory = '''<ul class="toc"><li><a href="commands.html">Commands</a><ul>
    <li><a href="times.html">times</a><ul><li><a href="commands.html#plus">Commands again</a></li></ul></li>
    <li><a href="minus.html">minus</a></li></ul></li>
    <li><a href="times.html">times again</a></li></ul>'''
    workspace, _ = _workspace(tmp_path, directory, ["minus", "times", "commands"])
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    assert _children(plan, "document") == ["commands"]
    commands = next(node for node in plan.nodes if node.title == "commands")
    assert _children(plan, commands.node_key) == ["times", "minus"]
    assert len(plan.nodes) == 4
    engine.materialize(plan)


def test_fragment_must_resolve_to_the_actual_heading_not_just_a_matching_label(tmp_path):
    directory = _directory("dl").replace("graphs.html#syntax", "graphs.html#missing-anchor")
    workspace, _ = _workspace(tmp_path, directory, ["features", "graphs"])
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    graphs = next(node for node in plan.nodes if node.title == "graphs")
    assert _children(plan, graphs.node_key) == ["features"]
    assert not any(node.title == "Syntax" for node in plan.nodes)
    assert engine.validate(plan).valid


def test_ambiguous_repeated_headings_stay_on_their_original_page(tmp_path):
    workspace, _ = _workspace(tmp_path, _directory("dl"), ["features", "graphs"],
        extra_body={"graphs": '<h2 id="other-syntax">Syntax</h2><p>Another context.</p>'})
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    graphs = next(node for node in plan.nodes if node.title == "graphs")
    assert _children(plan, graphs.node_key) == ["features"]
    assert not any(node.title == "Syntax" for node in plan.nodes)
    assert engine.validate(plan).valid


def test_prose_definitions_and_site_navigation_do_not_assign_topic_parents(tmp_path):
    directory = '''<header><ul class="toc"><li><a href="times.html">Site menu</a></li></ul></header>
    <dl><dt><p>Read about <a href="commands.html">commands</a> in context.</p></dt>
    <dd><p>Related prose refers to <a href="minus.html">subtraction</a>.</p></dd></dl>'''
    workspace, result = _workspace(tmp_path, directory, ["minus", "times", "commands"])
    structure = workspace.store.read_json_artifact(result["root_structure_artifact_id"])
    assert all(tree["items"][0]["label"] != "Site menu" for tree in structure["nested_link_trees"])
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    assert not any(node.title == "commands" and _children(plan, node.node_key) for node in plan.nodes)
    assert engine.validate(plan).valid
