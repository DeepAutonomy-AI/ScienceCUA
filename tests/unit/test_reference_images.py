import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent_export import export_agent_entries
from app.knowledge import validate_references
from app.models import EvidenceExcerpt, ReferenceCandidate, ReferenceImageRef, TargetProfile
from app.reference_images import hydrate_reference_images
from app.result_preview import read_result_preview


SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M0 0h24v24H0z"/></svg>'
IMAGE_ID = 'art_' + hashlib.sha256(SVG).hexdigest()[:24]


def image_catalog():
    text = 'Controls:\n| Function | Description |\n| --- | --- |\n| [image I15] | Adds a row after the current row |\n'
    evidence = EvidenceExcerpt(
        evidence_id='E1', source_id='SRC1', source_part_id='part_table', text_artifact_id='art_' + 'c' * 24,
        start_offset=100, end_offset=100 + len(text), content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        kind='html_section', title='Table controls', content=text, locator={}, extraction_method='structured',
        image_refs=[dict(artifact_id=IMAGE_ID, source_part_id='part_table', media_type='image/svg+xml',
                         page=1, anchor_offset=100 + text.index('[image I15]'))],
    )
    row = '| [image I15] | Adds a row after the current row |'
    reference = ReferenceCandidate(
        reference_id='R1', corpus_snapshot_id='CP1', title='Table controls',
        description='Table icons and their effects. Consult when changing a table.',
        reference_mode='source_catalog', source_catalog_id='C1',
        citations=[dict(evidence_id='E1', quote=text)],
        entries=[dict(entry_id='RE1', title='[image I15]', content=row, source_evidence_id='E1',
                      citations=[dict(evidence_id='E1', quote=row)])],
    )
    return reference, evidence


def test_validation_preserves_image_keyed_catalog_and_source_text():
    reference, evidence = image_catalog()
    original = reference.model_dump_json()
    accepted, checks = validate_references([reference], {'E1': evidence})
    assert checks[0].accepted, checks[0].errors
    entry = accepted[0].entries[0]
    assert entry.image_refs[0].artifact_id == IMAGE_ID
    assert entry.image_refs[0].marker == '[image I15]'
    assert entry.display_title == 'Adds a row after the current row'
    assert (entry.title, entry.content) == (reference.entries[0].title, reference.entries[0].content)
    assert reference.model_dump_json() == original


def test_page_local_image_numbers_do_not_leak_between_entries():
    reference, evidence = image_catalog()
    other_id = 'art_' + 'b' * 24
    other = evidence.model_copy(update={'evidence_id': 'E2', 'image_refs': [
        evidence.image_refs[0].model_copy(update={'artifact_id': other_id})]})
    second = reference.entries[0].model_copy(update={
        'entry_id': 'RE2', 'source_evidence_id': 'E2',
        'citations': [reference.entries[0].citations[0].model_copy(update={'evidence_id': 'E2'})]})
    reference = reference.model_copy(update={'entries': [*reference.entries, second]})
    hydrated = hydrate_reference_images(reference, {'E1': evidence, 'E2': other})
    assert [[im.artifact_id for im in e.image_refs] for e in hydrated.entries] == [[IMAGE_ID], [other_id]]


def test_ambiguous_summary_anchors_are_not_assigned_to_the_wrong_page():
    reference, evidence = image_catalog()
    other = evidence.model_copy(update={'evidence_id': 'E2', 'image_refs': [
        evidence.image_refs[0].model_copy(update={'artifact_id': 'art_' + 'b' * 24})]})
    reference = reference.model_copy(update={'reference_mode': 'summary', 'entries': [],
        'content': 'Two pages use [image I15].', 'citations': [*reference.citations,
            reference.citations[0].model_copy(update={'evidence_id': 'E2'})]})
    hydrated = hydrate_reference_images(reference, {'E1': evidence, 'E2': other})
    assert len(hydrated.image_refs) == 2
    assert all(not image.marker for image in hydrated.image_refs)


def test_reference_image_cannot_belong_to_unrelated_evidence():
    reference, evidence = image_catalog()
    forged = ReferenceImageRef(artifact_id='art_' + 'b' * 24, evidence_id='E1', media_type='image/svg+xml')
    reference = reference.model_copy(update={'image_refs': [forged]})
    accepted, checks = validate_references([reference], {'E1': evidence})
    assert not accepted
    assert any('not attached' in error for error in checks[0].errors)


def test_entry_does_not_inherit_other_images_from_the_same_evidence():
    reference, evidence = image_catalog()
    other = evidence.image_refs[0].model_copy(update={'artifact_id': 'art_' + 'b' * 24,
                                                    'anchor_offset': 100 + len(evidence.content) + 1})
    evidence = evidence.model_copy(update={
        'content': evidence.content + '\n[image I16]', 'image_refs': [*evidence.image_refs, other]})
    entry = hydrate_reference_images(reference, {'E1': evidence}).entries[0]
    assert [im.artifact_id for im in entry.image_refs] == [IMAGE_ID]


def test_unmarked_source_image_is_kept_with_its_exact_text_span():
    reference, evidence = image_catalog()
    text = 'The diagram shows the table layout.'
    evidence = evidence.model_copy(update={'content': text, 'start_offset': 0, 'image_refs': [
        evidence.image_refs[0].model_copy(update={'anchor_offset': 5})]})
    entry = reference.entries[0].model_copy(update={'title': 'Table layout', 'content': text})
    reference = reference.model_copy(update={'entries': [entry]})
    entry = hydrate_reference_images(reference, {'E1': evidence}).entries[0]
    assert [im.artifact_id for im in entry.image_refs] == [IMAGE_ID]
    assert entry.image_refs[0].marker == ''


@pytest.mark.parametrize('phase', ['completed', 'degraded'])
def test_saved_result_preview_is_text_only_without_rewriting_evidence_or_artifacts(phase):
    reference, evidence = image_catalog()
    original = {'snapshot': {'evidence': [evidence.model_dump(mode='json')]},
                'references': [reference.model_dump(mode='json')],
                'knowledge_snapshot': {'references': [reference.model_dump(mode='json')]}}
    before = json.dumps(original, sort_keys=True)
    preview = read_result_preview(SimpleNamespace(read_result=lambda: original), {'phase': phase})
    for container in (preview, preview['knowledge_snapshot']):
        rendered = json.dumps(container['references'])
        assert 'image_refs' not in rendered and '[image' not in rendered
        assert 'Adds a row after the current row' in rendered
    assert preview['snapshot'] == original['snapshot']
    assert json.dumps(original, sort_keys=True) == before


def test_export_preserves_reference_text_without_image_files_or_placeholders(tmp_path):
    reference, evidence = image_catalog()
    result = export_agent_entries(tmp_path / 'portable', run_id='run_test',
        target_profile=TargetProfile(software_name='Editor', target_version='1', platform='General'), skills=[],
        references=[reference], evidence_by_id={'E1': evidence},
        image_reader=lambda aid: pytest.fail('Text-only Reference export read an image'))
    content_path = Path(result['agent_reference_entries'][0]['content_file'])
    text = content_path.read_text('utf-8')
    assert '## Adds a row after the current row' in text
    assert 'Adds a row after the current row' in text
    image_links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    assert image_links == []
    assert '[image' not in text and 'images/' not in text
    assert not (content_path.parent / 'images').exists()
    assert '/api/' not in text and 'http:' not in text


def test_text_only_reference_export_does_not_require_an_image_reader(tmp_path):
    reference, evidence = image_catalog()
    destination = tmp_path / 'unwritten'
    result = export_agent_entries(destination, run_id='run_test',
        target_profile=TargetProfile(software_name='Editor', target_version='1', platform='General'), skills=[], references=[reference],
        evidence_by_id={'E1': evidence})
    content = Path(result['agent_reference_entries'][0]['content_file']).read_text('utf-8')
    assert '[image' not in content and 'Adds a row' in content


def test_standalone_reference_export_is_also_text_only(tmp_path):
    from app.knowledge import SkillRegistry
    reference, evidence = image_catalog()
    path = tmp_path / 'reference.md'
    SkillRegistry._write_agent_reference_markdown(path, reference,
        evidence_by_id={'E1': evidence}, image_reader=lambda aid: SVG)
    text = path.read_text('utf-8')
    assert 'Adds a row' in text and 'images/' not in text and '[image' not in text
    assert not (tmp_path / 'images').exists()


@pytest.mark.parametrize('stage', ['task_trajectory', 'trajectory_inference'])
def test_model_reference_lookup_tools_return_only_text(stage):
    from app.knowledge import KnowledgeWorkspace
    reference, evidence = image_catalog()
    workspace = KnowledgeWorkspace.__new__(KnowledgeWorkspace)
    workspace.stage = stage
    workspace.references_by_id = {'R1': hydrate_reference_images(reference, {'E1': evidence})}
    workspace.submitted_references = []
    workspace.evidence_by_id = {'E1': evidence}
    workspace.trajectory_packet_full_evidence_ids = set()
    before = workspace.references_by_id['R1'].model_dump_json()
    responses = [workspace.read_reference('R1'), workspace.read_reference_entry('R1', 'RE1'),
                 workspace.search_reference_entries('R1', 'Adds a row')]
    for result in responses:
        serialized = json.dumps(result)
        assert 'image_refs' not in serialized and '[image' not in serialized and IMAGE_ID not in serialized
        assert 'Adds a row' in serialized
    assert workspace.references_by_id['R1'].model_dump_json() == before
    assert evidence.image_refs[0].artifact_id == IMAGE_ID


def test_custom_reference_renderer_cannot_reintroduce_images_and_skills_are_unchanged(tmp_path):
    from test_agent_entry_export import _skill
    reference, evidence = image_catalog()
    skill = _skill()
    before = reference.model_dump_json()
    result = export_agent_entries(tmp_path / 'custom', run_id='run_test',
        target_profile=TargetProfile(software_name='Editor', target_version='1', platform='General'),
        skills=[skill], references=[reference], evidence_by_id={'E1': evidence},
        reference_body_renderer=lambda _: 'Keep this text. [image I15] ![icon](images/old.svg) <img src="old.png">')
    reference_content = Path(result['agent_reference_entries'][0]['content_file']).read_text('utf-8')
    assert reference_content.strip() == 'Keep this text.'
    skill_text = Path(result['agent_skill_entries'][0]['entry_file']).read_text('utf-8')
    assert skill.procedure[0].instruction in skill_text and skill.preconditions in skill_text
    assert reference.model_dump_json() == before
