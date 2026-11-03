import json
import re
from pathlib import Path

import pytest

from app.agent_export import AGENT_ENTRY_SCHEMA_VERSION, export_agent_entries
from app.models import (
    CorpusSnapshot,
    EvidenceCitation,
    EvidenceExcerpt,
    KnowledgeSnapshot,
    ReferenceCandidate,
    ReferenceEntry,
    SkillCandidate,
    SkillProcedureStepDraft,
    TargetProfile,
)


def _profile(**overrides):
    return TargetProfile(
        **{
            "software_name": "GRASS GIS",
            "software_id": "GRASS_GIS",
            "target_version": "8.4",
            "platform": "Windows",
            **overrides,
        }
    )


def _citation():
    return EvidenceCitation(evidence_id="E1", quote="Original documented capability")


def _skill(skill_id="SK1", title="Import georeferenced data", **overrides):
    return SkillCandidate(
        **{
            "skill_id": skill_id,
            "knowledge_snapshot_id": "knowledge",
            "title": title,
            "description": "Import georeferenced data. Use when creating a dataset.",
            "applicability": "Raster and vector datasets in the current project.",
            "preconditions": "A project with its CRS already selected.\nInput files are available.",
            "procedure": [
                SkillProcedureStepDraft(
                    ordinal=1,
                    instruction="Select the dataset and perform the documented import.",
                    citations=[_citation()],
                )
            ],
            "trajectory_ids": ["TR1"],
            **overrides,
        }
    )


def _reference(reference_id="R1", title="Vector command catalog", **overrides):
    description = overrides.pop("description", "GRASS GIS vector commands and capabilities; consult when identifying or checking a command during any task.")
    return ReferenceCandidate(
        **{
            "reference_id": reference_id,
            "corpus_snapshot_id": "corpus",
            "title": title,
            "description": description,
            "content": "Original overview content, including all details.",
            "reference_mode": "source_catalog",
            "injection_policy": "on_demand",
            "source_catalog_id": "catalog-original",
            "usage_tags": ["command_catalog"],
            "entries": [
                ReferenceEntry(
                    entry_id="RE1",
                    title="v.import",
                    content="Name: v.import\nParameters: input, output\nExample: v.import input=source output=target\n",
                    source_evidence_id="E1",
                    locator={"page": 17, "heading": "Vector commands"},
                    citations=[_citation()],
                )
            ],
            "citations": [_citation()],
            **overrides,
        }
    ).model_copy(update={"description": description})


def _frontmatter(path: Path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    header, body = text[4:].split("\n---\n", 1)
    # The exporter deliberately uses the JSON-string subset of YAML. Parsing
    # each scalar tests escaping without adding a YAML dependency to the app.
    return {
        key: json.loads(value)
        for key, value in (line.split(": ", 1) for line in header.splitlines())
    }, body


def _export(destination, skills=(), references=(), **overrides):
    return export_agent_entries(
        destination,
        run_id="run-one",
        target_profile=_profile(),
        skills=skills,
        references=references,
        **overrides,
    )


def _presentation(content):
    import hashlib
    return {
        'mode': 'agent_formatted', 'content': content,
        'source_artifact_id': 'art_' + 'a' * 24,
        'source_sha256': 'b' * 64,
        'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'evidence_ids': ['E1'],
    }


def test_formatted_reference_export_uses_presentation_without_duplicate_source_fragments(tmp_path):
    content = '## Operators\n\n| Operator | Alias | Precedence |\n| --- | --- | --- |\n| ** | ^ | 2 |\n'
    reference = _reference(presentation=_presentation(content))
    original = reference.model_dump_json()
    result = _export(tmp_path, references=[reference],
                     reference_body_renderer=lambda _: 'Historical rendering should not replace the presentation')
    path = Path(result['agent_reference_entries'][0]['content_file'])
    exported = path.read_text(encoding='utf-8')
    assert content.strip() in exported
    assert 'Original overview content' not in exported
    assert 'Name: v.import' not in exported
    assert 'Historical rendering' not in exported
    assert reference.model_dump_json() == original


def test_formatted_reference_preserves_literal_image_syntax_in_code_and_typed_presentation(tmp_path):
    from app.models import ReferencePresentation
    from app.reference_text import reference_presentation_content, text_only_reference
    content = '```text\n  ![literal](example.svg)\n  [image I1]\n  <img src="literal">\n```'
    reference = _reference(presentation=_presentation(content))
    projected = text_only_reference(reference)
    assert isinstance(projected.presentation, ReferencePresentation)
    assert projected.presentation.content == content
    assert reference_presentation_content(projected) == content
    assert reference_presentation_content(projected.model_dump(mode='json')) == content
    result = _export(tmp_path, references=[reference])
    assert content in Path(result['agent_reference_entries'][0]['content_file']).read_text(encoding='utf-8')


def test_source_table_cells_distinguish_literal_markup_from_layout_breaks():
    from app.reference_layout import _cell
    assert _cell('\\') == '\\\\'
    assert _cell('||') == '\\|\\|'
    assert _cell('a\\|b') == 'a\\\\\\|b'
    assert _cell('`x`') == '\\`x\\`'
    assert _cell('<br>\nnext') == '&lt;br&gt;<br>next'
    assert _cell('&lt;') == '&amp;lt;'


def test_mixed_entries_have_independent_yaml_entrypoints_and_linked_index(tmp_path):
    skill, reference = _skill(), _reference()
    result = _export(tmp_path, [skill], [reference])
    manifest = json.loads(Path(result["agent_entry_manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == AGENT_ENTRY_SCHEMA_VERSION
    assert (manifest["skill_count"], manifest["reference_count"], manifest["entry_count"]) == (1, 1, 2)
    assert (manifest["run_id"], manifest["software_id"], manifest["target_version"]) == ("run-one", "GRASS_GIS", "8.4")
    assert result["agent_entry_count"] == len(result["agent_entries"]) == 2
    index = Path(result["agent_entry_index_file"]).read_text(encoding="utf-8")
    assert not (tmp_path / "SKILL.md").exists()
    for record in manifest["entries"]:
        path = tmp_path / record["entry_path"]
        metadata, _ = _frontmatter(path)
        assert metadata == {"name": path.parent.name, "description": record["description"]}
        assert record["entry_path"] in index
        assert record["name"] in index
        assert record["description"] in index
        assert path.parent.parent == tmp_path / "entries"
    assert len(result["agent_skill_entries"]) == len(result["agent_reference_entries"]) == 1


def test_skill_procedure_and_required_preconditions_remain_self_contained(tmp_path):
    skill = _skill()
    result = _export(tmp_path, [skill])
    _, body = _frontmatter(Path(result["agent_skill_entries"][0]["entry_file"]))
    for original in (skill.description, skill.applicability, skill.preconditions, skill.procedure[0].instruction):
        assert original in body
    assert "## References" not in body
    assert "content.md" not in body


def test_historical_skill_caveats_are_preserved_when_supplied(tmp_path):
    caveats = ["Confirm the source coordinate system.", "Path note: Follow the installed software version."]
    skill = _skill().model_copy(update={"caveats": caveats})
    result = _export(tmp_path, [skill])
    _, body = _frontmatter(Path(result["agent_skill_entries"][0]["entry_file"]))
    assert "## Caveats" in body
    assert all(caveat in body for caveat in caveats)


def test_explicitly_empty_preconditions_export_no_prerequisites(tmp_path):
    result = _export(tmp_path, [_skill(preconditions="")])
    _, body = _frontmatter(Path(result["agent_skill_entries"][0]["entry_file"]))
    assert "## Preconditions" in body
    assert "No source-backed precondition was required by this Skill." in body


def test_reference_only_export_preserves_knowledge_without_audit_metadata(tmp_path):
    original = "Original overview\r\n  Preserve indentation.\r\n\r\n"
    reference = _reference(content=original)
    entry_text = "Name: v.import\r\nParameters: input, output\r\nExample: an exact example  \r\n\r\n"
    reference = reference.model_copy(update={
        "entries": [reference.entries[0].model_copy(update={"content": entry_text})]
    })
    result = _export(tmp_path, references=[reference])
    record = result["agent_reference_entries"][0]
    assert result["agent_skill_count"] == 0
    assert result["agent_reference_count"] == result["agent_entry_count"] == 1
    _, entry_body = _frontmatter(Path(record["entry_file"]))
    assert "[the complete reference](content.md)" in entry_body
    assert "during any task" in entry_body
    assert "Original overview" not in entry_body
    content = Path(record["content_file"]).read_bytes()
    assert original.encode("utf-8") in content
    assert entry_text.encode("utf-8") in content
    for metadata in ("RE1", "R1", "E1", "catalog-original", "command_catalog", "Vector commands",
                     "Original documented capability", "Locator:"):
        assert metadata.encode("utf-8") not in content
    assert b"Rationale:" not in content
    assert [line for line in entry_body.splitlines() if line.startswith("## ")] == [
        "## Knowledge scope", "## On-demand lookup",
    ]
    assert [line for line in content.decode("utf-8").splitlines() if line.startswith("## ")] == [
        "## v.import",
    ]
    assert (Path(record["entry_file"]).parent / "content.md").is_file()
    from app.knowledge import SkillRegistry

    legacy_path = tmp_path / "legacy-reference.md"
    SkillRegistry._write_agent_reference_markdown(legacy_path, reference)
    assert legacy_path.read_bytes() == content


def test_names_are_ascii_bounded_unique_and_stable_across_candidate_ids(tmp_path):
    skills = [_skill("SK1", "导入资料"), _skill("SK2", "导入资料"),
              _skill("SK3", "Import" * 40, description="First version"),
              _skill("SK4", "Import" * 40, description="Second version")]
    refs = [_reference("R1", "导入资料")]
    result = _export(tmp_path / "first", skills, refs)
    names = [item["name"] for item in result["agent_entries"]]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) for name in names)
    assert all(len(name) <= 64 for name in names)
    assert all(name.startswith("grass-gis-") for name in names)
    renamed_skills = [item.model_copy(update={"skill_id": f"SK{index + 10}"}) for index, item in enumerate(skills)]
    second = _export(tmp_path / "second", renamed_skills, [refs[0].model_copy(update={"reference_id": "R10"})])
    assert names == [item["name"] for item in second["agent_entries"]]


def test_all_chinese_software_gets_safe_deterministic_names(tmp_path):
    result = export_agent_entries(tmp_path, run_id="run-chinese", target_profile=_profile(software_id=None, software_name="地理信息软件"),
                                  skills=[_skill(title="导入资料")], references=[])
    name = result["agent_entries"][0]["name"]
    assert re.fullmatch(r"software-[a-f0-9]+-skill-entry-[a-f0-9]+", name)
    assert len(name) <= 64


def test_frontmatter_roundtrips_quotes_unicode_colons_and_multiline_description(tmp_path):
    description = 'Lookup "yes": paths C:\\data\\catalog\n查询命令；[scope] | parameters.'
    result = _export(tmp_path, references=[_reference(title="Catalog [commands] | lookups", description=description)])
    header, _ = _frontmatter(Path(result["agent_entries"][0]["entry_file"]))
    assert header["description"] == description
    index = Path(result["agent_entry_index_file"]).read_text(encoding="utf-8")
    assert "\\|" in index
    assert "\\[commands\\]" in index
    assert len([line for line in index.splitlines() if line.startswith("| reference |")]) == 1


def test_long_skill_description_uses_concise_metadata_and_preserves_full_body(tmp_path):
    description = "Import georeferenced raster or vector data into a project. " + "Additional documented applicability and prerequisites. " * 55
    skill = _skill(description=description)
    result = _export(tmp_path, skills=[skill])
    metadata, body = _frontmatter(Path(result["agent_skill_entries"][0]["entry_file"]))
    assert len(metadata["description"]) <= 1_024
    assert metadata["description"].startswith("Import georeferenced")
    assert metadata["description"].endswith("...")
    assert description in body
    manifest = json.loads(Path(result["agent_entry_manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["entries"][0]["description"] == metadata["description"]


def test_overlong_reference_description_is_rejected_without_summarizing(tmp_path):
    reference = _reference().model_copy(update={"description": "Knowledge scope. " * 100})
    destination = tmp_path / "unwritten"
    with pytest.raises(ValueError, match="exceeds the native 1024-character"):
        _export(destination, references=[reference])
    assert not destination.exists()


@pytest.mark.parametrize("description", [None, "", " \n\t"])
def test_missing_reference_description_rejects_whole_export_before_writes(tmp_path, description):
    destination = tmp_path / "unwritten"
    reference = _reference().model_copy(update={"description": description})
    with pytest.raises(ValueError, match="requires an authored description"):
        _export(destination, skills=[_skill()], references=[reference])
    assert not destination.exists()


@pytest.mark.parametrize("language", ["English", "Chinese"])
def test_unstructured_reference_exports_lookup_sections_and_original_entry(tmp_path, language):
    reference = _reference()
    reference = reference.model_copy(update={"entries": [reference.entries[0].model_copy(update={"content": "Performs a vector import."})]})
    result = _export(tmp_path, references=[reference], skill_output_language=language)
    record = result["agent_reference_entries"][0]
    _, body = _frontmatter(Path(record["entry_file"]))
    content = Path(record["content_file"]).read_text(encoding="utf-8")
    expected_entry_headings = (
        ["## 知识范围", "## 按需查阅"] if language == "Chinese"
        else ["## Knowledge scope", "## On-demand lookup"]
    )
    expected_content_headings = ["## v.import"]
    assert [line for line in body.splitlines() if line.startswith("## ")] == expected_entry_headings
    assert [line for line in content.splitlines() if line.startswith("## ")] == expected_content_headings
    assert reference.entries[0].content in content


def test_summary_reference_uses_knowledge_body(tmp_path):
    reference = _reference(reference_mode="summary", source_catalog_id=None, entries=[],
                           content="A complete source-backed conceptual explanation.")
    result = _export(tmp_path, references=[reference])
    record = result["agent_reference_entries"][0]
    _, body = _frontmatter(Path(record["entry_file"]))
    assert [line for line in body.splitlines() if line.startswith("## ")] == [
        "## Knowledge scope", "## On-demand lookup",
    ]
    content = Path(record["content_file"]).read_text(encoding="utf-8")
    assert reference.content in content
    assert reference.description in content
    assert content == f"# {reference.title}\n\n{reference.description}\n\n{reference.content}\n"


def test_chinese_skill_labels_preserve_existing_output_language(tmp_path):
    skill = _skill().model_copy(update={"caveats": ["确认资料中的坐标系。"]})
    result = _export(tmp_path, skills=[skill], skill_output_language="Chinese")
    _, body = _frontmatter(Path(result["agent_skill_entries"][0]["entry_file"]))
    for label in ("## 适用范围", "## 前置条件", "## 操作步骤"):
        assert label in body
    assert "预期观察结果" not in body
    assert "## 注意事项" in body
    assert "确认资料中的坐标系。" in body


def test_optional_renderers_preserve_existing_body_and_complete_reference(tmp_path):
    skill_body = "# Existing skill\r\nOriginal steps and preconditions.\r\n"
    reference_body = "# Existing complete reference\r\nAll source knowledge.\r\n"
    result = _export(tmp_path, [_skill()], [_reference()],
                     skill_body_renderer=lambda _: skill_body,
                     reference_body_renderer=lambda _: reference_body)
    assert Path(result["agent_skill_entries"][0]["entry_file"]).read_bytes().endswith(skill_body.encode("utf-8"))
    assert Path(result["agent_reference_entries"][0]["content_file"]).read_bytes() == reference_body.encode("utf-8")
    assert not list(tmp_path.rglob("*.tmp"))


def test_empty_export_still_writes_manifest_and_human_index(tmp_path):
    result = _export(tmp_path)
    assert result["agent_entry_count"] == 0
    assert Path(result["agent_entry_directory"]).is_dir()
    manifest = json.loads(Path(result["agent_entry_manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["entries"] == []
    assert "No entries were generated." in Path(result["agent_entry_index_file"]).read_text(encoding="utf-8")


def test_reexport_of_same_entries_updates_body_without_stale_directories(tmp_path):
    first = _export(tmp_path, skills=[_skill()])
    skill = _skill(preconditions="A newly clarified prerequisite.")
    second = _export(tmp_path, skills=[skill])
    assert first["agent_entries"][0]["name"] == second["agent_entries"][0]["name"]
    assert skill.preconditions in Path(second["agent_entries"][0]["entry_file"]).read_text(encoding="utf-8")
    assert len(list((tmp_path / "entries").glob("*/SKILL.md"))) == 1


def test_reexport_with_changed_entry_set_rejects_without_deleting_or_changing_manifest(tmp_path):
    first = _export(tmp_path, skills=[_skill()])
    manifest_path = Path(first["agent_entry_manifest_file"])
    original_manifest = manifest_path.read_bytes()
    original_entry = Path(first["agent_entries"][0]["entry_file"])
    original_body = original_entry.read_bytes()
    with pytest.raises(ValueError, match="different set of entries"):
        _export(tmp_path, skills=[_skill(title="A different operation")])
    assert manifest_path.read_bytes() == original_manifest
    assert original_entry.read_bytes() == original_body
    assert len(list((tmp_path / "entries").glob("*/SKILL.md"))) == 1


def test_preexisting_unmanaged_discoverable_entry_is_not_silently_left_in_export(tmp_path):
    unmanaged = tmp_path / "entries" / "previous-entry" / "SKILL.md"
    unmanaged.parent.mkdir(parents=True)
    unmanaged.write_text("Existing artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this export"):
        _export(tmp_path, skills=[_skill()])
    assert unmanaged.read_text(encoding="utf-8") == "Existing artifact"
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("with_skill", [True, False], ids=["mixed", "reference-only"])
def test_skill_registry_commit_exports_peer_entries_and_separate_reference_description(tmp_path, with_skill):
    from app.knowledge import SkillRegistry

    profile = _profile()
    quote = _citation().quote
    evidence = EvidenceExcerpt(
        evidence_id="E1", source_id="SRC1", source_part_id="manual-vector-commands",
        kind="reference", title="Vector commands", content=quote,
        start_offset=0, end_offset=len(quote), extraction_method="test",
        text_artifact_id="art_000000000000000000000001", content_sha256="1" * 64,
        locator={"source_origin": "official", "source_resolved_version": "8.4", "page": 17},
        resolved_version="8.4",
    )
    corpus = CorpusSnapshot(
        corpus_snapshot_id="corpus", run_id="run-registry", target_profile=profile,
        source_results=[], evidence=[evidence], evidence_count=1,
        source_part_count=1, source_status_counts={"succeeded": 1},
    )
    original_content = "Full source overview\r\n  Original indentation and details.\r\n\r\n"
    reference = _reference(content=original_content)
    knowledge = KnowledgeSnapshot(
        knowledge_snapshot_id="knowledge", run_id="run-registry", corpus_snapshot_id="corpus",
        corpus_snapshot_artifact_id="art_000000000000000000000002", target_profile=profile,
        tasks=[], references=[reference],
    )
    registry = SkillRegistry(tmp_path / "output")
    result = registry.commit(
        run_id="run-registry", target_profile=profile, corpus=corpus, knowledge=knowledge,
        trajectories=[], skills=[_skill()] if with_skill else [], validations=[],
    )

    entry_root = Path(result["agent_entry_directory"])
    assert entry_root == tmp_path / "output" / "GRASS_GIS" / "run-registry" / "entries"
    run_root = entry_root.parent
    assert Path(result["output_directory"]) == run_root
    assert Path(result["audit_directory"]) == run_root / "audit"
    assert Path(result["skills_file"]) == run_root / "audit" / "skills.json"
    assert (run_root / "audit" / "README.md").is_file()
    assert result["agent_skill_directory"] == result["agent_reference_directory"] == str(entry_root)
    assert result["agent_skill_index_file"] == result["agent_reference_index_file"] == result["agent_entry_index_file"]
    assert Path(result["agent_entry_index_file"]).parent == entry_root.parent
    assert (result["skill_count"], result["reference_count"], result["agent_entry_count"]) == (int(with_skill), 1, int(with_skill) + 1)
    assert len(list(entry_root.glob("*/SKILL.md"))) == result["agent_entry_count"]
    assert not (entry_root.parent / "skills").exists()
    for item in result["agent_entries"]:
        entry_path = Path(item["entry_file"])
        assert entry_path.parent.parent == entry_root
        metadata, _ = _frontmatter(entry_path)
        assert metadata["name"] == item["name"] == entry_path.parent.name

    audit = json.loads(Path(result["skills_file"]).read_text(encoding="utf-8"))
    assert audit["schema_version"] == "v8"
    saved_reference = audit["references"][0]
    assert saved_reference["description"] == reference.description
    assert saved_reference["content"] == original_content
    assert saved_reference["entries"][0]["content"] == reference.entries[0].content
    assert saved_reference["reference_id"] == reference.reference_id
    assert saved_reference["source_catalog_id"] == reference.source_catalog_id
    assert saved_reference["entries"][0]["entry_id"] == reference.entries[0].entry_id
    assert saved_reference["entries"][0]["source_evidence_id"] == reference.entries[0].source_evidence_id
    assert saved_reference["entries"][0]["locator"] == reference.entries[0].locator
    assert "caveats" not in saved_reference
    assert audit["evidence_manifest"][0]["source_resolved_version"] == "8.4"
    assert audit["evidence_manifest"][0]["source_origin"] == "official"
    assert len(audit["skills"]) == int(with_skill)
    if with_skill:
        saved_skill = audit["skills"][0]
        assert "preconditions" in saved_skill
        assert "rationale" not in saved_skill["procedure"][0]["citations"][0]
    assert "rationale" not in saved_reference["citations"][0]
    reference_entry = result["agent_reference_entries"][0]
    metadata, entry_body = _frontmatter(Path(reference_entry["entry_file"]))
    assert metadata["description"] == reference.description
    assert reference.description in entry_body
    assert original_content not in entry_body
    exported_content = Path(reference_entry["content_file"]).read_bytes()
    assert original_content.encode("utf-8") in exported_content
    assert reference.entries[0].content.encode("utf-8") in exported_content
    assert reference.description.encode("utf-8") in exported_content
    assert b"source=official; version=8.4" not in exported_content
    assert b"Original documented capability" not in exported_content
    if with_skill:
        skill_body = Path(result["agent_skill_entries"][0]["entry_file"]).read_text(encoding="utf-8")
        assert _skill().preconditions in skill_body
        assert "## References" not in skill_body
    else:
        assert result["agent_skill_entries"] == []



@pytest.mark.parametrize("language", ["English", "Chinese"])
def test_skill_markdown_exports_drop_retired_observations(tmp_path, language):
    from app.knowledge import SkillRegistry
    step = _skill().procedure[0].model_dump(mode="json")
    step["expected_observation"] = "Retired observation text."
    skill = _skill(procedure=[step])
    result = _export(tmp_path / "entries", [skill], skill_output_language=language)
    paths = [Path(result["agent_skill_entries"][0]["entry_file"])]
    for name, renderer in [("audit", SkillRegistry._write_skill_markdown),
                           ("agent", SkillRegistry._write_agent_skill_markdown)]:
        path = tmp_path / (name + ".md")
        renderer(path, skill, skill_output_language=language)
        paths.append(path)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert step["instruction"] in text
        assert "Retired observation text." not in text
        assert "Expected observation" not in text
        assert "\u9884\u671f\u89c2\u5bdf\u7ed3\u679c" not in text
