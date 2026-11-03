from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError
from task_model_helpers import first_level_tree
from tests.knowledge_fixtures import trajectory_draft

from app.knowledge import (
    _knowledge_tool_catalog,
    _validate_citations,
    build_pipeline_quality_report,
    validate_references,
    validate_skills,
    validate_tasks,
    validate_trajectories,
)
from app.models import (
    CorpusSnapshot, EvidenceCitation, EvidenceExcerpt, EvidenceImageRef, KnowledgeSnapshot, ReferenceCandidate,
    SkillCandidate, SkillDraft, SkillProcedureStepDraft, TaskCandidate,
    TargetProfile, TrajectoryCandidate,
)


def _evidence(content='Press Ctrl+R\n\nto render the graph.'):
    return EvidenceExcerpt(
        evidence_id='E1', source_id='SRC1', source_part_id='SP1',
        text_artifact_id='art_test', start_offset=0, end_offset=len(content),
        content_sha256='test', kind='html_section', title='Rendering controls',
        content=content, locator={}, extraction_method='test',
    )




@pytest.mark.parametrize('quote', [None, '', ' \n\t'])
def test_rationale_cannot_replace_a_nonempty_text_quote(quote):
    payload = {'evidence_id': 'E1', 'rationale': 'This Evidence supports rendering.'}
    if quote is not None:
        payload['quote'] = quote
    with pytest.raises(ValidationError, match='nonempty quote'):
        EvidenceCitation.model_validate(payload)


def test_unknown_evidence_and_unattached_images_still_fail():
    evidence = _evidence()
    unknown = EvidenceCitation(evidence_id='E2', quote='Supporting text.')
    assert _validate_citations([unknown], {'E1': evidence}) == ['Unknown Evidence ID: E2']
    image_id = 'art_' + '1' * 24
    visual = EvidenceCitation(evidence_id='E1', image_artifact_id=image_id)
    assert 'not attached to Evidence E1' in _validate_citations([visual], {'E1': evidence})[0]
    evidence = evidence.model_copy(update={'image_refs': [EvidenceImageRef(
        artifact_id=image_id, source_part_id='SP1', media_type='image/png', page=1,
    )]})
    assert _validate_citations([visual], {'E1': evidence}) == []


def test_all_four_downstream_stages_accept_a_formatted_quote():
    evidence = _evidence()
    evidence_by_id = {'E1': evidence}
    citation = EvidenceCitation(evidence_id='E1', quote='Press Ctrl+R to render the graph.')
    task = TaskCandidate(task_id='T1', corpus_snapshot_id='CP1', task='Render a graph', citations=[citation], category='Operations')
    reference = ReferenceCandidate(
        reference_id='R1', corpus_snapshot_id='CP1', title='Graph rendering shortcut',
        content='Ctrl+R renders the current graph.', citations=[citation],
    )
    trajectory = TrajectoryCandidate(
        trajectory_id='TR1', knowledge_snapshot_id='KP1',
        **trajectory_draft(task_id=task.task_id, instruction=citation.quote,
                           expected_observation='The graph is rendered.',
                           citations=[citation.model_dump(mode='json')]),
    )
    skill = SkillCandidate(
        skill_id='SK1', knowledge_snapshot_id='KP1', title='Render the current graph',
        description='Render the graph using its documented keyboard shortcut.',
        applicability='Use when the current graph is ready for rendering.',
        preconditions='', trajectory_ids=['TR1'],
        procedure=[SkillProcedureStepDraft(
            ordinal=1, instruction='Press Ctrl+R to render the graph.',
            citations=[citation],
        )],
    )
    outcomes = [
        validate_tasks([task], evidence_by_id, [first_level_tree(evidence)]),
        validate_references([reference], evidence_by_id),
        validate_trajectories([trajectory], evidence_by_id, {'T1': task}),
        validate_skills([skill], evidence_by_id, {'T1': task}, {'TR1': trajectory}),
    ]
    for accepted, validation in outcomes:
        assert len(accepted) == 1, validation
        assert validation[0].accepted, validation[0].errors


def test_documentation_wording_does_not_reject_trajectory_or_skill():
    instruction = "For command-line or scripted workflows, use the `dockprep` command; the documentation identifies it as Dock Prep's command implementation."
    content = 'Dock Prep prepares structures. The command implementation is dockprep.'
    evidence_by_id = {'E1': _evidence(content)}
    citation = EvidenceCitation(evidence_id='E1', quote=content)
    task = TaskCandidate(task_id='T1', corpus_snapshot_id='CP1', task='Prepare structures', citations=[citation], category='Operations')
    trajectory = TrajectoryCandidate(trajectory_id='TR1', knowledge_snapshot_id='KP1', **trajectory_draft(task_id=task.task_id, instruction=instruction, expected_observation='The selected structures are prepared.', citations=[citation.model_dump(mode='json')]))
    skill = SkillCandidate(skill_id='SK1', knowledge_snapshot_id='KP1', title=task.task, description=task.task, applicability='Use when structures need preparation.', preconditions='', trajectory_ids=['TR1'], procedure=[SkillProcedureStepDraft(ordinal=1, instruction=instruction, citations=[citation])])
    outcomes = [validate_trajectories([trajectory], evidence_by_id, {'T1': task}), validate_skills([skill], evidence_by_id, {'T1': task}, {'TR1': trajectory})]
    for accepted, validation in outcomes:
        assert len(accepted) == 1, validation
        assert validation[0].accepted, validation[0].errors


def _skill_without_top_level_citations():
    return {
        'title': 'Render the current graph',
        'description': 'Render the graph using its documented control.',
        'applicability': 'Use when a graph is ready for rendering.',
        'preconditions': '', 'trajectory_ids': ['TR1'],
        'procedure': [{
            'ordinal': 1, 'instruction': 'Press Ctrl+R to render the graph.',
            'citations': [{'evidence_id': 'E1', 'quote': 'Press Ctrl+R to render the graph.'}],
        }],
    }


def _skill_linkage():
    citation = EvidenceCitation(evidence_id='E1', quote='Press Ctrl+R to render the graph.')
    task = TaskCandidate(task_id='T1', corpus_snapshot_id='CP1', task='Render a graph', citations=[citation], category='Operations')
    trajectory = TrajectoryCandidate(
        trajectory_id='TR1', knowledge_snapshot_id='KP1',
        **trajectory_draft(task_id=task.task_id, instruction=citation.quote,
                           expected_observation='The graph is rendered.',
                           citations=[citation.model_dump(mode='json')]),
    )
    return {task.task_id: task}, {trajectory.trajectory_id: trajectory}






def test_retired_skill_fields_are_rejected_by_new_drafts_and_stripped_from_saved_skills():
    payload = _skill_without_top_level_citations()
    retired = {
        'reference_ids': ['R1'], 'reference_trigger_type': 'required', 'task_ids': ['T1'],
        'caveats': ['Requires an open graph.'], 'confidence': 0.9, 'inferred': True,
        'completion_criteria': [{'criterion': 'The graph is rendered.'}],
        'citations': copy.deepcopy(payload['procedure'][0]['citations']),
    }
    for field, value in retired.items():
        with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
            SkillDraft.model_validate({**payload, field: value})
    payload.update(retired)
    before = copy.deepcopy(payload)
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    assert not retired.keys() & saved.model_dump().keys()
    assert payload == before
    assert saved.trajectory_ids == ['TR1']
    assert saved.procedure[0].citations[0].evidence_id == 'E1'


def test_skill_schema_contains_only_the_simplified_fields():
    schema = _knowledge_tool_catalog()['submit_skill_batch']['function']['parameters']['properties']['skills']['items']
    expected = {'title', 'description', 'applicability', 'preconditions', 'procedure', 'trajectory_ids'}
    assert set(schema['properties']) == expected
    assert set(schema['required']) == expected
    assert set(SkillDraft.model_fields) == expected
    assert set(SkillCandidate.model_fields) == expected | {
        'skill_id', 'knowledge_snapshot_id', 'status', 'created_at',
    }
    assert schema['properties']['preconditions']['type'] == 'string'
    assert 'citations' in schema['properties']['procedure']['items']['required']


def test_skill_task_validation_is_derived_from_trajectory():
    skill = SkillCandidate.model_validate({
        **_skill_without_top_level_citations(), 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1',
    })
    tasks, trajectories = _skill_linkage()
    accepted, validation = validate_skills([skill], {'E1': _evidence()}, tasks, trajectories)
    assert len(accepted) == 1, validation
    assert accepted[0].skill_id == skill.skill_id
    assert validation[0].accepted
    assert 'task_ids' not in accepted[0].model_dump()
    trajectories['TR1'] = trajectories['TR1'].model_copy(update={'task_id': 'T2'})
    accepted, validation = validate_skills([skill], {'E1': _evidence()}, tasks, trajectories)
    assert not accepted
    assert any('T2' in error for error in validation[0].errors)


def test_skill_quality_task_coverage_is_derived_and_deduplicated_through_trajectories():
    tasks, trajectory_map = _skill_linkage()
    task_one = tasks['T1']
    task_two = task_one.model_copy(update={'task_id': 'T2', 'title': 'Render another graph'})
    task_three = task_one.model_copy(update={'task_id': 'T3', 'title': 'Uncovered operation'})
    trajectory_one = trajectory_map['TR1']
    trajectory_two = trajectory_one.model_copy(update={'trajectory_id': 'TR2'})
    trajectory_three = trajectory_one.model_copy(update={'trajectory_id': 'TR3', 'task_id': 'T2'})
    skill = SkillCandidate.model_validate({
        **_skill_without_top_level_citations(), 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1',
        'trajectory_ids': ['TR1', 'TR2', 'TR3'],
    })
    target = TargetProfile(software_name='Example', target_version='1', platform='General', ui_locale='English')
    corpus = CorpusSnapshot(
        corpus_snapshot_id='CP1', run_id='run_test', target_profile=target,
        evidence=[_evidence()], evidence_count=1, source_part_count=1,
        source_results=[], source_status_counts={},
    )
    knowledge = KnowledgeSnapshot(
        knowledge_snapshot_id='KP1', run_id=corpus.run_id, corpus_snapshot_id=corpus.corpus_snapshot_id,
        corpus_snapshot_artifact_id='art_corpus', target_profile=target,
        tasks=[task_one, task_two, task_three],
    )
    quality = build_pipeline_quality_report(
        corpus, knowledge, [trajectory_one, trajectory_two, trajectory_three], [skill], [],
    )
    assert quality['linkage'] == {
        'tasks_without_skill': ['T3'], 'trajectories_without_skill': [],
        'unknown_task_links': [], 'unknown_trajectory_links': [],
    }
    assert quality['skill_contract']['skills_linking_multiple_tasks'] == 1
    singleton = skill.model_copy(update={'trajectory_ids': ['TR1', 'TR2']})
    quality = build_pipeline_quality_report(
        corpus, knowledge, [trajectory_one, trajectory_two, trajectory_three], [singleton], [],
    )
    assert quality['linkage']['tasks_without_skill'] == ['T2', 'T3']
    assert quality['skill_contract']['skills_linking_multiple_tasks'] == 0


def test_citation_rationale_is_removed_from_global_model_and_tool_schema():
    payload = {'evidence_id': 'E1', 'quote': 'Supporting excerpt.', 'rationale': 'Legacy explanation.'}
    citation = EvidenceCitation.model_validate(payload)
    assert 'rationale' not in citation.model_dump()
    assert 'rationale' not in EvidenceCitation.model_fields
    assert payload['rationale'] == 'Legacy explanation.'
    schema = _knowledge_tool_catalog()['submit_skill_batch']['function']['parameters']['properties']['skills']['items']['properties']['procedure']['items']['properties']['citations']['items']
    assert set(schema['properties']) == {'evidence_id', 'quote', 'image_artifact_id'}
    assert schema['additionalProperties'] is False


def test_legacy_citation_rationale_is_stripped_in_saved_skill_and_other_downstream_models():
    payload = _skill_without_top_level_citations()
    payload['procedure'][0]['citations'][0]['rationale'] = 'Legacy skill explanation.'
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    assert 'rationale' not in saved.model_dump()['procedure'][0]['citations'][0]
    assert payload['procedure'][0]['citations'][0]['rationale'] == 'Legacy skill explanation.'
    tasks, trajectories = _skill_linkage()
    reference = ReferenceCandidate(
        reference_id='R1', corpus_snapshot_id='CP1', title='Graph rendering shortcut',
        content='Ctrl+R renders the current graph.', citations=tasks['T1'].citations,
    )
    for item in (tasks['T1'], trajectories['TR1'], reference):
        historical = item.model_dump()
        historical['citations'][0]['rationale'] = 'Legacy explanation.'
        restored = type(item).model_validate(historical)
        assert 'rationale' not in restored.model_dump()['citations'][0]
        assert historical['citations'][0]['rationale'] == 'Legacy explanation.'


def test_skill_model_and_tool_require_only_nested_citations():
    payload = _skill_without_top_level_citations()
    draft = SkillDraft.model_validate(payload)
    assert 'citations' not in draft.model_dump()
    assert 'citations' not in SkillDraft.model_json_schema()['properties']
    schema = _knowledge_tool_catalog()['submit_skill_batch']['function']['parameters']['properties']['skills']['items']
    assert 'citations' not in schema['properties']
    assert 'citations' not in schema['required']
    assert 'citations' in schema['properties']['procedure']['items']['required']

    del payload['procedure'][0]['citations']
    with pytest.raises(ValidationError, match='citations'):
        SkillDraft.model_validate(payload)
    payload['procedure'][0]['citations'] = []
    with pytest.raises(ValidationError, match='at least 1 item'):
        SkillDraft.model_validate(payload)




def test_skill_step_citations_still_validate_evidence_identity():
    payload = _skill_without_top_level_citations()
    payload['procedure'][0]['citations'][0]['evidence_id'] = 'E2'
    skill = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    accepted, validation = validate_skills([skill], {'E1': _evidence()}, *_skill_linkage())
    assert not accepted
    assert 'Procedure step 1: Unknown Evidence ID: E2' in validation[0].errors


def test_skill_preconditions_are_one_required_string():
    payload = _skill_without_top_level_citations()
    payload['preconditions'] = 'The graph view is open and the current graph is ready to render.'
    draft = SkillDraft.model_validate(payload)
    assert draft.model_dump()['preconditions'] == payload['preconditions']
    schema = _knowledge_tool_catalog()['submit_skill_batch']['function']['parameters']['properties']['skills']['items']
    precondition_schema = schema['properties']['preconditions']
    assert precondition_schema['type'] == 'string'
    assert 'preconditions' in schema['required']
    assert 'preconditions' in SkillDraft.model_json_schema()['required']
    assert 'criterion' not in schema['properties']
    skill = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    accepted, validation = validate_skills([skill], {'E1': _evidence()}, *_skill_linkage())
    assert len(accepted) == 1, validation[0].errors

    payload['preconditions'] = ''
    assert SkillDraft.model_validate(payload).preconditions == ''
    del payload['preconditions']
    with pytest.raises(ValidationError, match='preconditions'):
        SkillDraft.model_validate(payload)
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    assert saved.preconditions == ''
    assert 'preconditions' not in payload
    payload['preconditions'] = [{'criterion': 'The graph view is open.'}]
    with pytest.raises(ValidationError, match='valid string'):
        SkillDraft.model_validate(payload)


def test_saved_precondition_citations_are_removed_without_changing_other_citations():
    payload = _skill_without_top_level_citations()
    citations = list(payload['procedure'][0]['citations'])
    payload['preconditions'] = [
        {'criterion': 'The graph view is open.', 'citations': citations},
        {'criterion': 'The current graph is ready to render.'},
    ]
    payload['completion_criteria'] = [{'criterion': 'The graph is rendered.', 'citations': citations}]
    with pytest.raises(ValidationError, match='valid string'):
        SkillDraft.model_validate(payload)
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    dumped = saved.model_dump(mode='json')
    assert dumped['preconditions'] == 'The graph view is open.\nThe current graph is ready to render.'
    normalized_citations = [EvidenceCitation.model_validate(item).model_dump(mode='json') for item in citations]
    assert dumped['procedure'][0]['citations'] == normalized_citations
    assert 'completion_criteria' not in dumped
    assert payload['preconditions'][0]['citations'] == citations


def test_saved_empty_preconditions_become_an_empty_string():
    payload = _skill_without_top_level_citations()
    payload['preconditions'] = []
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    assert saved.preconditions == ''
    assert payload['preconditions'] == []


def test_tool_schema_and_model_agree_on_quote_or_image_presence():
    jsonschema = pytest.importorskip('jsonschema')
    parameters = _knowledge_tool_catalog()['submit_trajectory']['function']['parameters']
    schema = parameters['properties']['steps']['items']['properties']['citations']['items']
    examples = [
        ({'evidence_id': 'E1', 'quote': 'Supporting excerpt.'}, True),
        ({'evidence_id': 'E1', 'rationale': 'Supporting explanation.'}, False),
        ({'evidence_id': 'E1', 'quote': ''}, False),
        ({'evidence_id': 'E1', 'quote': ' \n\t'}, False),
        ({'evidence_id': 'E1', 'image_artifact_id': 'art_' + '1' * 24}, True),
        ({'evidence_id': 'E1', 'quote': '', 'image_artifact_id': 'art_' + '1' * 24}, True),
        ({'evidence_id': 'E1', 'image_artifact_id': None}, False),
    ]
    for payload, valid in examples:
        assert jsonschema.Draft202012Validator(schema).is_valid(payload) == valid, payload
        if valid:
            EvidenceCitation.model_validate(payload)
        else:
            with pytest.raises(ValidationError):
                EvidenceCitation.model_validate(payload)



def test_skill_steps_remove_observations_from_history_and_reject_them_in_new_drafts():
    payload = _skill_without_top_level_citations()
    expected_fields = {'ordinal', 'instruction', 'citations'}
    assert set(SkillProcedureStepDraft.model_fields) == expected_fields
    draft = SkillDraft.model_validate(payload)
    assert set(draft.model_dump()['procedure'][0]) == expected_fields
    payload['procedure'][0]['expected_observation'] = 'Retired observation text.'
    before = copy.deepcopy(payload)
    with pytest.raises(ValidationError, match='expected_observation'):
        SkillDraft.model_validate(payload)
    saved = SkillCandidate.model_validate({**payload, 'skill_id': 'SK1', 'knowledge_snapshot_id': 'KP1'})
    assert set(saved.model_dump()['procedure'][0]) == expected_fields
    assert payload == before
    tools = _knowledge_tool_catalog()
    skill_step = tools['submit_skill_batch']['function']['parameters']['properties']['skills']['items']['properties']['procedure']['items']
    assert set(skill_step['properties']) == set(skill_step['required']) == expected_fields
    assert 'expected_observation' in tools['submit_trajectory']['function']['parameters']['properties']['steps']['items']['properties']
    assert _skill_linkage()[1]['TR1'].steps[0].expected_observation == 'The graph is rendered.'
