from __future__ import annotations

import copy
import hashlib
import json

import fitz
import pytest

from app.models import EvidenceExcerpt, EvidenceImageRef, SourcePart, SourceResult
from app.orchestration.source_stage import validate_source_results
from app.storage import RunStore


def image_bytes(shade=255):
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False)
    pixmap.clear_with(shade)
    return pixmap.tobytes('png')


def source_fixture(tmp_path, *, standalone=False):
    """Persist real image bytes and exact text, with an unaffected text Evidence."""
    store = RunStore(tmp_path / 'data', 'run_image_validation')
    text = 'Choose File > Export and inspect the diagram.'
    text_record = store.put_artifact(text.encode(), kind='source_text', media_type='text/plain',
                                     suffix='.txt', source_id='SRC1')
    raw = store.put_artifact(b'<h1>Export</h1>', kind='raw_source', media_type='text/html',
                             suffix='.html', source_id='SRC1')
    image = store.put_artifact(image_bytes(), kind='source_image', media_type='image/png',
                               suffix='.png', source_id='SRC1')
    other_image = store.put_artifact(image_bytes(0), kind='source_image', media_type='image/png',
                                     suffix='.png', source_id='SRC1')
    part = SourcePart(part_id='P1', source_id='SRC1', kind='pdf_block' if standalone else 'html_section',
                      label='Export', locator={'page': 1}, artifact_id=raw.artifact_id,
                      text_artifact_id=text_record.artifact_id, char_count=len(text),
                      content_sha256=text_record.sha256,
                      metadata={} if standalone else {'images': {'I1': {
                          'status': 'saved', 'artifact_id': image.artifact_id, 'media_type': 'image/png'}}})
    image_part = SourcePart(part_id='Pimage', source_id='SRC1', kind='pdf_image', label='Diagram',
                            locator={'page': 1}, artifact_id=image.artifact_id)
    other_part = SourcePart(part_id='P2', source_id='SRC1', kind='html_section', label='Other section',
                            locator={'page': 1}, artifact_id=raw.artifact_id,
                            metadata={'images': {'I2': {'status': 'saved',
                                'artifact_id': other_image.artifact_id, 'media_type': 'image/png'}}})
    parts = [part.model_dump(mode='json'), other_part.model_dump(mode='json')]
    if standalone:
        parts.append(image_part.model_dump(mode='json'))
    store.write_source_json('SRC1', 'parts.json', parts)
    corpus = store.put_json_artifact({'source_id': 'SRC1', 'parts': parts}, kind='source_corpus',
                                     source_id='SRC1')
    evidence = EvidenceExcerpt(evidence_id='E1', source_id='SRC1', source_part_id='P1',
                               text_artifact_id=text_record.artifact_id, start_offset=0,
                               end_offset=len(text), content_sha256=text_record.sha256,
                               kind='instruction', title='Export diagram', content=text,
                               locator={'page': 1}, extraction_method='source_text_span',
                               image_refs=[EvidenceImageRef(artifact_id=image.artifact_id,
                                   source_part_id='Pimage' if standalone else 'P1', media_type='image/png',
                                   page=1)])
    unaffected = evidence.model_copy(deep=True)
    unaffected.evidence_id = 'E2'
    unaffected.image_refs = []
    result = SourceResult(source_id='SRC1', display_name='Export manual',
                          modality='pdf' if standalone else 'html', status='succeeded',
                          evidence=[evidence, unaffected], source_part_count=len(parts),
                          corpus_artifact_id=corpus.artifact_id)
    return store, result, parts


@pytest.mark.parametrize('standalone', [False, True], ids=['html-section-attachment', 'pdf-image-part'])
def test_owned_saved_image_preserves_evidence(tmp_path, standalone):
    store, result, _ = source_fixture(tmp_path, standalone=standalone)
    original = copy.deepcopy(result.model_dump(mode='json')['evidence'])
    accepted, summaries = validate_source_results(store, [result])
    assert accepted[0].status == 'succeeded'
    assert [item.model_dump(mode='json') for item in accepted[0].evidence] == original
    assert summaries[0]['errors'] == []
    assert summaries[0]['rejected_evidence_count'] == 0
    events, _ = store.read_events()
    event = next(event for event in events if event['event_type'] == 'source_validation_completed')
    assert event['level'] == 'info'
    assert event['payload']['input_evidence_count'] == event['payload']['evidence_count'] == 2
    assert event['payload']['error_count'] == 0


def test_existing_image_from_another_html_section_is_rejected(tmp_path):
    store, result, parts = source_fixture(tmp_path)
    ref = result.evidence[0].image_refs[0]
    ref.artifact_id = parts[1]['metadata']['images']['I2']['artifact_id']
    assert store.artifact_path(ref.artifact_id).is_file()
    accepted, summaries = validate_source_results(store, [result])
    assert [item.evidence_id for item in accepted[0].evidence] == ['E2']
    assert accepted[0].status == 'degraded'
    assert summaries[0]['rejected_evidence_count'] == 1
    assert any('image' in error.lower() for error in summaries[0]['errors'])


def test_source_with_only_invalid_image_evidence_fails_and_logs_error(tmp_path):
    store, result, parts = source_fixture(tmp_path)
    result.evidence = result.evidence[:1]
    parts[0]['metadata']['images'] = {}
    store.write_source_json('SRC1', 'parts.json', parts)
    accepted, summaries = validate_source_results(store, [result])
    assert accepted[0].status == 'failed'
    assert accepted[0].evidence == []
    assert 'branch_validation_failed' in accepted[0].gaps
    assert summaries[0]['input_evidence_count'] == 1
    assert summaries[0]['evidence_count'] == 0
    assert summaries[0]['rejected_evidence_count'] == 1
    events, _ = store.read_events()
    event = next(event for event in events if event['event_type'] == 'source_validation_completed')
    assert event['level'] == 'error'
    assert event['payload']['error_preview']


@pytest.mark.parametrize('fault', [
    'unknown_part', 'part_other_source', 'artifact_other_source', 'not_image',
    'ref_media_type', 'registered_media_type', 'not_saved', 'unregistered',
    'missing_file', 'changed_file', 'wrong_byte_count',
])
def test_invalid_html_image_is_quarantined_without_losing_other_evidence(tmp_path, fault):
    store, result, parts = source_fixture(tmp_path)
    ref = result.evidence[0].image_refs[0]
    registration = parts[0]['metadata']['images']['I1']
    metadata_path = store.root / 'artifacts' / 'meta' / f'{ref.artifact_id}.json'
    if fault == 'unknown_part':
        ref.source_part_id = 'unknown'
    elif fault == 'part_other_source':
        parts[0]['source_id'] = 'SRC2'
    elif fault in {'artifact_other_source', 'not_image', 'wrong_byte_count'}:
        record = json.loads(metadata_path.read_text(encoding='utf-8'))
        if fault == 'artifact_other_source':
            record['source_id'] = 'SRC2'
        elif fault == 'not_image':
            record['media_type'] = 'text/plain'
        else:
            record['byte_count'] += 1
        metadata_path.write_text(json.dumps(record), encoding='utf-8')
    elif fault == 'ref_media_type':
        ref.media_type = 'image/jpeg'
    elif fault == 'registered_media_type':
        registration['media_type'] = 'image/jpeg'
    elif fault == 'not_saved':
        registration['status'] = 'missing'
    elif fault == 'unregistered':
        parts[0]['metadata']['images'] = {}
    elif fault == 'missing_file':
        store.artifact_path(ref.artifact_id).unlink()
    else:
        path = store.artifact_path(ref.artifact_id)
        # Keep byte count unchanged so this independently exercises hash validation.
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    store.write_source_json('SRC1', 'parts.json', parts)
    accepted, summaries = validate_source_results(store, [result])
    assert accepted[0].status == 'degraded'
    assert [item.evidence_id for item in accepted[0].evidence] == ['E2']
    assert 'evidence_validation_rejected:1' in accepted[0].gaps
    assert summaries[0]['rejected_evidence_count'] == 1
    assert summaries[0]['errors']


@pytest.mark.parametrize('fault', ['missing_file', 'changed_file', 'ref_media_type'])
def test_standalone_image_keeps_integrity_and_type_checks(tmp_path, fault):
    store, result, _ = source_fixture(tmp_path, standalone=True)
    ref = result.evidence[0].image_refs[0]
    if fault == 'missing_file':
        store.artifact_path(ref.artifact_id).unlink()
    elif fault == 'ref_media_type':
        ref.media_type = 'image/jpeg'
    else:
        path = store.artifact_path(ref.artifact_id)
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    accepted, summaries = validate_source_results(store, [result])
    assert [item.evidence_id for item in accepted[0].evidence] == ['E2']
    assert summaries[0]['rejected_evidence_count'] == 1


def test_source_validation_audit_links_full_detail_after_rejections(tmp_path):
    store, result, parts = source_fixture(tmp_path)
    parts[0]['metadata']['images'] = {}
    store.write_source_json('SRC1', 'parts.json', parts)
    rejected = []
    for index in range(27):
        evidence = result.evidence[0].model_copy(deep=True)
        evidence.evidence_id = f'Ebad{index}'
        rejected.append(evidence)
    result.evidence = [*rejected, result.evidence[1]]
    accepted, summaries = validate_source_results(store, [result])
    events, _ = store.read_events(limit=2000)
    validation_events = [event for event in events if event['event_type'] == 'source_validation_completed']
    assert len(validation_events) == 1
    event = validation_events[0]
    assert event['source_id'] == 'SRC1'
    assert event['level'] == 'warning'
    payload = event['payload']
    assert payload['input_evidence_count'] == 28
    assert payload['evidence_count'] == 1
    assert payload['rejected_evidence_count'] == 27
    assert payload['error_count'] == len(summaries[0]['errors'])
    assert len(payload['rejected_evidence_ids']) == 20
    assert len(payload['error_preview']) == 5
    detail_id = payload['validation_artifact_id']
    record = store.artifact_record(detail_id)
    assert record.kind == 'source_validation_detail'
    assert record.source_id == 'SRC1'
    detail = store.read_json_artifact(detail_id)
    assert detail['schema_version'] == 'source-validation-detail.v1'
    assert detail['input_evidence_count'] == 28
    assert detail['evidence_count'] == len(accepted[0].evidence) == 1
    assert detail['rejected_evidence_count'] == 27
    assert detail['error_count'] == len(detail['errors']) == len(summaries[0]['errors'])
    assert detail['errors'] == summaries[0]['errors']
    assert detail['rejected_evidence_ids'] == [evidence.evidence_id for evidence in rejected]
    assert len(detail['rejected_evidence']) == 27
    assert all(entry['evidence_id'] and entry['errors'] for entry in detail['rejected_evidence'])
    assert detail['fatal_errors'] == []
    assert hashlib.sha256(store.artifact_path(detail_id).read_bytes()).hexdigest() == record.sha256
