"""Preserve authored linear fallbacks of CSS-rendered mathematical fractions."""
import pytest

from app.html_guided.extractor import GuidedExtractor
from app.html_sourcepart.extractor import Extractor, all_blocks, all_runs


CSS = '.ignored, .ordinary-hidden { display:none } .denominator { border-top:1px solid }'
FRACTION = ('<span class="formula">4<i>π</i> × <span class="fraction">'
            '<span class="ignored">(</span><span class="numerator">[<i>Area</i>]</span>'
            '<span class="ignored">)/(</span>'
            '<span class="denominator">[<i>Perimeter</i>]<sup>2</sup></span>'
            '<span class="ignored">)</span></span></span>')


def extract(markup, *, css=CSS, cls=GuidedExtractor):
    source = f'<html><head><title>Manual</title></head><body><main><h1>Measurements</h1>{markup}</main></body></html>'
    return cls('https://example.test/manual.html', stylesheets=[css]).extract(source)


def content(document):
    return '\n\n'.join(part['content'] for part in document['parts'])


def visible_runs(document):
    return [run for section in document['sections'] for run in all_runs(section['blocks'])]


def test_guided_fraction_restores_only_original_linear_operator_and_provenance():
    document = extract(f'<p>Circularity = {FRACTION}.</p>'
                       '<p>Visible<span class="ordinary-hidden">Private note</span>end.</p>')
    assert document['extraction']['mode'] == 'structured'
    assert not document['extraction']['degraded']
    text = content(document)
    assert '( [*Area*]' not in text
    assert '([*Area*])/([*Perimeter*]<sup>2</sup>)' in text
    assert 'Private note' not in text
    assert any('Private note' in ''.join(run.get('text', '') for run in entry['runs'])
               for entry in document['annotations'].values())
    record, = document['guided_math_fractions']
    assert record['status'] == 'authored_fallback_restored'
    assert [token['text'] for token in record['restored_tokens']] == ['(', ')/(', ')']
    assert all(token['dom_path'].startswith(record['dom_path']) for token in record['restored_tokens'])
    assert 'class="ignored"' in record['raw_markup']
    assert all(run.get('dom_path') for run in visible_runs(document))
    # The original captured source is immutable and still declares hidden text.
    assert '<span class="ignored">)/(</span>' in document['source_snapshot']['content']


@pytest.mark.parametrize('wrapper', ['<span hidden>{}</span>', '<span style="display:none">{}</span>',
                                    '<span class="whole-hidden">{}</span>'])
def test_hidden_equation_does_not_expose_its_fallback(wrapper):
    document = extract('<p>Before ' + wrapper.format(FRACTION) + ' after.</p>',
                       css=CSS + '.whole-hidden {display:none}')
    assert 'Area' not in content(document)
    assert not document.get('guided_math_fractions')
    assert document['annotations']


@pytest.mark.parametrize('tag', ['pre', 'code', 'textarea'])
def test_math_like_markup_inside_code_is_unchanged(tag):
    markup = f'<{tag}>{FRACTION}</{tag}>'
    current = extract(markup)
    baseline = extract(markup, cls=Extractor)
    assert content(current) == content(baseline)
    assert not current.get('guided_math_fractions')


def test_escaped_html_code_sample_and_literal_indentation_remain_exact():
    literal = '  <span class="fraction">\n    <span class="numerator">x</span>\n  </span>'
    escaped = literal.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    document = extract('<pre>' + escaped + '</pre><p>' + FRACTION + '</p>')
    codes = [block for section in document['sections'] for block in all_blocks(section['blocks'])
             if block['type'] == 'code']
    assert ''.join(run.get('text', '') for run in codes[0]['runs']) == literal
    assert len(document['guided_math_fractions']) == 1


def test_fraction_classes_outside_mathematics_do_not_unhide_text():
    markup = FRACTION.replace('class="formula"', 'class="business-ratio"')
    document = extract('<p>' + markup + '</p>')
    assert ')/(' not in content(document)
    assert not document.get('guided_math_fractions')


@pytest.mark.parametrize('middle', ['', '<span hidden>divided by</span>', '<span hidden>)/</span>'])
def test_missing_or_ambiguous_fraction_fallback_is_explicitly_unsupported(middle):
    fraction = ('<span class="formula"><span class="fraction">'
                '<span class="numerator">a</span>' + middle +
                '<span class="denominator">b</span></span></span>')
    document = extract('<p>' + fraction + '</p>')
    assert document['extraction']['mode'] == 'structured'
    assert document['extraction']['degraded']
    assert any(reason['code'] == 'guided_math_fraction_unsupported'
               for reason in document['extraction']['reasons'])
    assert document['guided_math_fractions'][0]['status'] == 'unsupported'
    assert '/' not in content(document)
    assert all(part['metadata']['degraded'] for part in document['parts'])


def test_existing_hidden_slash_without_parentheses_is_restored():
    document = extract('<p><span class="math"><span class="fraction">'
                       '<span class="numerator">a</span><span hidden>/</span>'
                       '<span class="denominator">b</span></span></span></p>')
    assert 'a/b' in content(document)
    assert not document['extraction']['degraded']


@pytest.mark.parametrize('slash', ['⁄', '∕'])
def test_existing_visible_fraction_slash_is_preserved_without_false_unsupported(slash):
    document = extract('<p><span class="formula"><span class="fraction">'
                       '<sup class="numerator">1</sup>' + slash +
                       '<sub class="denominator">2</sub></span></span></p>')
    assert '<sup>1</sup>' + slash + '<sub>2</sub>' in content(document)
    assert not document['extraction']['degraded']
    assert document['guided_math_fractions'][0]['status'] == 'authored_fallback_visible'


def test_nested_fractions_restore_both_authored_fallbacks():
    inner = ('<span class="fraction"><span hidden>(</span><span class="numerator">a</span>'
             '<span hidden>)/(</span><span class="denominator">b</span><span hidden>)</span></span>')
    outer = ('<span class="formula"><span class="fraction"><span hidden>(</span>'
             '<span class="numerator">' + inner + '</span><span hidden>)/(</span>'
             '<span class="denominator">c</span><span hidden>)</span></span></span>')
    document = extract('<p>' + outer + '</p>')
    assert '((a)/(b))/(c)' in content(document)
    assert len(document['guided_math_fractions']) == 2


def test_shared_legacy_extractor_behavior_is_unchanged():
    document = extract('<p>' + FRACTION + '</p>', cls=Extractor)
    assert '([*Area*])/([*Perimeter*]' not in content(document)
    assert '[*Area*][*Perimeter*]' in content(document)
    assert not document.get('guided_math_fractions')


def test_unsupported_fraction_reaches_native_page_metadata_and_source_gaps(tmp_path):
    from app.acquisition import AcquisitionService
    from app.config import Settings
    from app.evidence_planning import EvidencePlanEngine
    from app.models import StagedInput, TargetProfile
    from app.storage import RunStore
    from app.toolkit import SourceWorkspace, ToolExecutor

    source = ('<html><head><title>Manual</title></head><body><main><h1>Measurements</h1>'
              '<p>Ratio <span class="formula"><span class="fraction">'
              '<span class="numerator">a</span><span class="denominator">b</span>'
              '</span></span>.</p></main></body></html>')
    path = tmp_path / 'manual.html'
    path.write_text(source, encoding='utf8')
    settings = Settings(data_dir=tmp_path / 'data', output_dir=tmp_path / 'out', max_html_pages=0)
    store = RunStore(settings.data_dir, 'run_html_fraction')
    target = TargetProfile(software_name='Example', target_version='1', platform='General', ui_locale='English')
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id='IN1', kind='uploaded_file', display_name='Manual',
                                               staged_path=str(path), declared_media_type='text/html')], target).manifests[0]
    manifest.source_metadata['html_processing_mode'] = 'model_guided'
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    executor = ToolExecutor(workspace, 'fraction_test')
    inventory = executor.call('html_inventory', {'max_pages': 0})
    assert inventory['ok']
    page, = inventory['pages']
    assert page['extraction']['degraded']
    assert page['extraction']['reasons'][0]['code'] == 'guided_math_fraction_unsupported'
    assert any('guided_math_fraction_unsupported' in gap for gap in workspace.gaps)
    engine = EvidencePlanEngine(workspace)
    plan = engine.build_skeleton_plan()
    result = executor._commit_materialized_evidence(
        plan, engine.materialize(plan), summary='Preserved unsupported fraction.', gaps=[], warnings=[],
        accepted_event='fraction_test_verified')
    assert result['accepted']
    assert workspace.final_result.status == 'degraded'
    assert any('guided_math_fraction_unsupported' in gap for gap in workspace.final_result.gaps)
