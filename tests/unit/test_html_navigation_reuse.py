"""Shared navigation decisions reuse exact observations, never URL semantics."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.html_guided.pipeline import build_snapshot, run_guided_html
from app.html_guided.reuse import navigation_signature, outside_content
from test_html_guided_pipeline import ROOT_URL, install_pages, plan_for, workspace_for


OTHER = "https://example.org/other-book/"
CHILD = "https://example.org/docs/lesson.html"
NAV = '<nav aria-label="Manual navigation"><ul><li><a href="/other-book/">Other book</a></li></ul></nav>'


def page(main: str, *, nav: str = NAV) -> str:
    return '<html><head><title>Example manual</title></head><body>' + nav + '<main>' + main + '</main></body></html>'


def observations(workspace):
    result = []
    for page_part in workspace.parts.values():
        if page_part.kind == 'html_page':
            value = workspace.store.read_json_artifact(page_part.metadata['link_observations_artifact_id'])
            result.extend(value['links'])
    return result


def signature_fixture():
    containers = [{'container_id': 'page-one-body', 'dom_path': 'html > body'},
                  {'container_id': 'page-one-nav', 'dom_path': 'html > body > nav'}]
    occurrence = {
        'label': 'Other book', 'context': 'Other book', 'heading_path': [],
        'disclosure_path': [], 'dom_path': 'html > body > nav > ul > li > a',
        'in_chrome': True, 'landmark_context': [{'tag': 'nav', 'role': '', 'label': 'Manual navigation'}],
        'authored_contents': False, 'in_code': False, 'page_navigation': None,
        'content_root_ids': ['page-one-nav', 'page-one-body'],
    }
    link = {'url': OTHER, 'href': OTHER, 'fragment': None, 'kind': 'document', 'fetchable': True,
            'legacy_role': 'directory_entry', 'occurrence_count': 1,
            'page_navigation_only': False, 'non_navigation_occurrence_count': 1,
            'occurrences': [occurrence]}
    return link, containers


def test_signature_uses_page_independent_container_paths_without_losing_observations():
    link, containers = signature_fixture()
    first = navigation_signature(link, containers)
    assert first
    second, second_containers = deepcopy(link), deepcopy(containers)
    for item in second_containers:
        item['container_id'] = item['container_id'].replace('one', 'two')
    second['occurrences'][0]['content_root_ids'] = [value.replace('one', 'two')
                                                   for value in second['occurrences'][0]['content_root_ids']]
    assert navigation_signature(second, second_containers) == first
    assert link == signature_fixture()[0]


def test_whole_navigation_changed_elsewhere_invalidates_an_unchanged_link():
    first, _ = build_snapshot(page('<h1>Lesson</h1><p>Run the analysis.</p>'), ROOT_URL)
    changed_nav = NAV.replace('</ul>', '<li><a href="/new-volume/">Another volume</a></li></ul>')
    second, _ = build_snapshot(page('<h1>Lesson</h1><p>Run the analysis.</p>', nav=changed_nav), ROOT_URL)
    link_a = next(link for link in first['links'] if link['url'] == OTHER)
    link_b = next(link for link in second['links'] if link['url'] == OTHER)
    assert link_a['context'] == link_b['context']
    assert navigation_signature(link_a, first['containers']) != navigation_signature(link_b, second['containers'])


def test_local_navigation_controls_do_not_change_the_identity_of_an_unchanged_catalog():
    source = page('<h1>Lesson</h1><p>Run.</p>', nav=NAV.replace('</nav>', '<a href="#" class="expander">Expand</a></nav>'))
    a, _ = build_snapshot(source, ROOT_URL)
    b, _ = build_snapshot(source, CHILD)
    la = next(link for link in a['links'] if link['url'] == OTHER)
    lb = next(link for link in b['links'] if link['url'] == OTHER)
    assert navigation_signature(la, a['containers']) == navigation_signature(lb, b['containers'])
    different, _ = build_snapshot(source.replace('href="#"', 'href="#another"'), CHILD)
    ld = next(link for link in different['links'] if link['url'] == OTHER)
    assert navigation_signature(ld, different['containers']) != navigation_signature(lb, b['containers'])
    based = source.replace('<head>', '<head><base href="https://example.org/foreign.html">')
    c, _ = build_snapshot(based, CHILD)
    lc = next(link for link in c['links'] if link['url'] == OTHER)
    assert navigation_signature(lc, c['containers']) != navigation_signature(lb, b['containers'])


@pytest.mark.asyncio
async def test_large_global_audit_references_complete_page_observations_without_copying_context(tmp_path, monkeypatch):
    links = ''.join(f'<li><a href="/volume-{index}/">Volume {index}</a></li>' for index in range(140))
    workspace = workspace_for(tmp_path, page('<h1>Manual</h1><p>Only this procedure.</p>',
                                           nav='<nav><ul>' + links + '</ul></nav>'))
    install_pages(monkeypatch, workspace, {})

    async def decide(payload):
        return plan_for(payload, action='exclude')

    outcome = await run_guided_html(workspace, decide)
    index = workspace.store.read_json_artifact(outcome['decision_artifact_id'])
    assert index['facts_storage'] == 'referenced_page_observations'
    assert len(index['decisions']) == 140
    for record in index['decisions']:
        assert 'occurrences' not in record and 'context' not in record
        page_records = workspace.store.read_json_artifact(record['observations_artifact_id'])['links']
        full = next(item for item in page_records if item['link_id'] == record['link_id'])
        assert full['url'] == record['url'] and full['occurrences']
        assert full['context']


@pytest.mark.parametrize('field,value', [
    ('label', 'Another book'), ('context', 'This reference is required for the exercise'),
    ('heading_path', ['Practical lesson']), ('disclosure_path', ['Advanced materials']),
    ('dom_path', 'html > body > nav > ul > li:nth-of-type(2) > a'),
    ('landmark_context', [{'tag': 'nav', 'role': '', 'label': 'Different manual'}]),
    ('page_navigation', {'relation': 'contents', 'basis': 'rel_attribute'}),
])
def test_signature_changes_when_the_observed_navigation_context_changes(field, value):
    link, containers = signature_fixture()
    changed = deepcopy(link)
    changed['occurrences'][0][field] = value
    assert navigation_signature(changed, containers) != navigation_signature(link, containers)


@pytest.mark.parametrize('field,value', [('in_chrome', False), ('authored_contents', True), ('in_code', True)])
def test_authored_body_contents_and_code_cannot_enter_navigation_cache(field, value):
    link, containers = signature_fixture()
    link['occurrences'][0][field] = value
    assert navigation_signature(link, containers) is None


def test_a_new_body_occurrence_prevents_reuse_even_if_the_url_and_sidebar_are_unchanged():
    link, containers = signature_fixture()
    body = deepcopy(link['occurrences'][0])
    body.update(in_chrome=False, context='Read this chapter before running the exercise',
                dom_path='html > body > main > p > a')
    link['occurrences'].append(body)
    link['occurrence_count'] = 2
    assert navigation_signature(link, containers) is None


def test_missing_container_and_external_navigation_cannot_be_treated_as_equal_dom_context():
    link, containers = signature_fixture()
    assert navigation_signature(link, containers[:1]) is None
    assert navigation_signature({**link, 'navigation_source': {'file': 'external-toc.json'}}, containers) is None
    assert navigation_signature({**link, 'external_navigation_entries': [{'label': 'Other book'}]}, containers) is None
    assert outside_content(link, 'main-root') is True
    assert outside_content(link, 'page-one-body') is False


@pytest.mark.asyncio
@pytest.mark.parametrize('use_default', [False, True])
async def test_identical_chrome_reuses_explicit_model_or_explicit_default_exclusion(tmp_path, monkeypatch, use_default):
    root = page('<h1>Example lessons</h1><ul><li><a href="lesson.html">Lesson</a></li></ul>')
    workspace = workspace_for(tmp_path, root)
    requested = install_pages(monkeypatch, workspace, {CHILD: page('<h1>Lesson</h1><p>Run the analysis.</p>')})
    calls = []

    async def decide(payload):
        offered = payload['page']['links']
        calls.append((payload['page']['url'], [item['url'] for item in offered]))
        plan = plan_for(payload, choose=lambda item: 'fetch' if item['url'] == CHILD else 'exclude')
        if use_default:
            plan['links'] = [item for item in plan['links'] if next(
                link['url'] for link in offered if link['link_id'] == item['link_id']) == CHILD]
            plan.update(default_action='exclude', default_reason='Other navigation belongs to a separate book.')
        return plan

    await run_guided_html(workspace, decide)
    assert requested == [CHILD]
    assert OTHER in calls[0][1]
    assert OTHER not in next(urls for url, urls in calls if url == CHILD)
    reused = [item for item in observations(workspace) if item.get('decision_origin') == 'reused_model']
    assert len(reused) == 1
    assert reused[0]['url'] == OTHER and reused[0]['source_url'] == CHILD
    assert reused[0]['action'] == 'exclude'
    assert reused[0]['reused_from']['source_url'] == ROOT_URL
    assert reused[0]['reused_from']['plan_artifact_id']
    assert reused[0]['reused_from']['snapshot_artifact_id']


@pytest.mark.asyncio
async def test_same_url_new_authored_body_use_returns_to_the_model(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, page('<h1>Lessons</h1><p><a href="lesson.html">Lesson</a></p>'))
    install_pages(monkeypatch, workspace, {CHILD: page('<h1>Lesson</h1><p>Discuss <a href="/other-book/">Other book</a> as a citation.</p>')})
    calls = []

    async def decide(payload):
        calls.append((payload['page']['url'], [item['url'] for item in payload['page']['links']]))
        return plan_for(payload, choose=lambda item: 'fetch' if item['url'] == CHILD else 'exclude')

    await run_guided_html(workspace, decide)
    assert OTHER in next(urls for url, urls in calls if url == CHILD)
    assert not any(item.get('decision_origin') == 'reused_model' for item in observations(workspace))


@pytest.mark.asyncio
async def test_selected_content_root_containing_cached_navigation_restores_model_decision(tmp_path, monkeypatch):
    workspace = workspace_for(tmp_path, page('<h1>Lessons</h1><p><a href="lesson.html">Lesson</a></p>'))
    install_pages(monkeypatch, workspace, {CHILD: page('<h1>Lesson</h1><p>Full page includes its document navigation.</p>')})
    calls = []

    async def decide(payload):
        calls.append((payload['page']['url'], [item['url'] for item in payload['page']['links']]))
        plan = plan_for(payload, choose=lambda item: 'fetch' if item['url'] == CHILD else 'exclude')
        if payload['page']['url'] == CHILD:
            plan['content_root_id'] = next(item['container_id'] for item in payload['page']['containers'] if item['tag'] == 'body')
        return plan

    await run_guided_html(workspace, decide)
    child_calls = [urls for url, urls in calls if url == CHILD]
    assert len(child_calls) == 2
    assert child_calls[0] == [] and child_calls[1] == [OTHER]
    records = [item for item in observations(workspace) if item['source_url'] == CHILD and item['url'] == OTHER]
    assert len(records) == 1 and records[0]['decision_origin'] == 'model'


@pytest.mark.asyncio
async def test_navigation_decisions_never_leak_between_acquisition_runs(tmp_path, monkeypatch):
    root = page('<h1>Lessons</h1><p>Independent entry with the same side navigation.</p>')
    offered = []
    for label in ('first-scope', 'second-scope'):
        workspace = workspace_for(tmp_path / label, root)
        workspace.manifest.origin_url = workspace.manifest.final_url = 'https://example.org/' + label + '/'
        install_pages(monkeypatch, workspace, {})

        async def decide(payload):
            offered.append([item['url'] for item in payload['page']['links']])
            return plan_for(payload, action='exclude')

        await run_guided_html(workspace, decide)
        assert not any(item.get('decision_origin') == 'reused_model' for item in observations(workspace))
    assert len(offered) == 2 and all(OTHER in urls for urls in offered)
