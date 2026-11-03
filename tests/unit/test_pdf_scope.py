"""Chapter selection preserves original PDF geometry without adjacent content."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
import pytest

from app.acquisition import AcquisitionService
from app.config import Settings
from app.models import StagedInput, TargetProfile
from app.pdf_sourcepart import extractor
from app.pdf_sourcepart.scope import preview_pdf_outline
from app.storage import RunStore
from app.toolkit import SourceWorkspace, ToolExecutor


@pytest.fixture
def chapter_pdf(tmp_path):
    path = tmp_path / 'chapters.pdf'
    with fitz.open() as document:
        page = document.new_page(width=500, height=700)
        page.insert_text((40, 60), 'Alpha', fontsize=22, fontname='hebo')
        page.insert_text((40, 100), 'ALPHA_PRIVATE_CONTENT belongs to Alpha.', fontsize=11)
        page.insert_text((40, 280), 'Beta', fontsize=22, fontname='hebo')
        page.insert_text((40, 320), 'BETA_CHOSEN_CONTENT belongs to Beta.', fontsize=11)
        page.insert_text((40, 400), 'Beta details', fontsize=17, fontname='hebo')
        page.insert_text((40, 440), 'BETA_DETAIL_CONTENT remains in its chapter.', fontsize=11)
        page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(40, 84, 250, 102),
                          'uri': 'https://example.org/alpha-private'})
        page.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(40, 304, 250, 322),
                          'uri': 'https://example.org/beta-chosen'})
        page = document.new_page(width=500, height=700)
        page.insert_text((40, 60), 'Gamma', fontsize=22, fontname='hebo')
        page.insert_text((40, 100), 'GAMMA_CONTENT belongs to Gamma.', fontsize=11)
        page = document.new_page(width=500, height=700)
        page.insert_text((40, 60), 'Delta', fontsize=22, fontname='hebo')
        page.insert_text((40, 100), 'DELTA_ORIGINAL_PAGE_THREE is retained.', fontsize=11)
        document.set_toc([[1, 'Alpha', 1], [1, 'Beta', 1], [2, 'Beta details', 1],
                          [1, 'Gamma', 2], [1, 'Delta', 3]])
        document.save(path)
    return path


def selection(preview, *titles):
    return {'mode': 'selected', 'sections': preview['sections'],
            'section_ids': [item['section_id'] for item in preview['sections'] if item['title'] in titles]}


def body(document):
    return '\n'.join(part['content'] for part in document['parts'])


def test_preview_is_lightweight_and_has_stable_exact_first_level_boundaries(chapter_pdf):
    with patch.object(extractor, 'extract_pdf', side_effect=AssertionError('full parse in preview')), \
            patch.object(fitz.Page, 'get_pixmap', side_effect=AssertionError('preview rendered a page')), \
            patch.object(fitz.Document, 'extract_image', side_effect=AssertionError('preview extracted an image')), \
            patch.object(fitz.Document, 'get_toc', side_effect=AssertionError('preview read native bookmarks')):
        preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
        again = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    assert preview['outline_available']
    assert preview['outline_source'] == 'layout_headings'
    assert preview['sections'] == again['sections']
    assert [item['title'] for item in preview['sections']] == ['Alpha', 'Beta', 'Gamma', 'Delta']
    alpha, beta = preview['sections'][:2]
    assert alpha['page_start'] == alpha['page_end'] == beta['page_start'] == 1
    assert alpha['end_anchor'] == beta['start_anchor']
    assert alpha['start_anchor']['y'] < alpha['end_anchor']['y']


def test_same_page_selection_keeps_original_text_links_and_cropped_visuals(chapter_pdf, tmp_path):
    preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    document = extractor.extract_pdf(chapter_pdf, tmp_path / 'selected', source_id='SRC1',
                                     extraction_scope=selection(preview, 'Beta'))
    text = body(document)
    assert 'BETA_CHOSEN_CONTENT' in text and 'BETA_DETAIL_CONTENT' in text
    assert 'ALPHA_PRIVATE_CONTENT' not in text and 'GAMMA_CONTENT' not in text
    assert document['outline'] == []
    assert {'Beta', 'Beta details'} <= {block.get('title') for page in document['pages']
                                     for block in page['blocks'] if block['type'] == 'heading'}
    assert 'beta-chosen' in json.dumps(document['links'])
    assert 'alpha-private' not in json.dumps(document['links'])
    first_page = document['pages'][0]
    assert first_page['page'] == 1
    assert all(asset['bbox'][1] >= preview['sections'][1]['start_anchor']['y']
               for asset in first_page['assets'])
    assert all(line['bbox'][1] >= preview['sections'][1]['start_anchor']['y']
               for line in json.loads((tmp_path / 'selected/raw/page-0001.json').read_text())['lines'])
    assert document['summary']['all_pages_represented']


def test_nonadjacent_selection_does_not_reintroduce_intervening_chapter(chapter_pdf, tmp_path):
    preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    document = extractor.extract_pdf(chapter_pdf, tmp_path / 'nonadjacent',
                                     extraction_scope=selection(preview, 'Alpha', 'Gamma'))
    assert 'ALPHA_PRIVATE_CONTENT' in body(document) and 'GAMMA_CONTENT' in body(document)
    assert 'BETA_CHOSEN_CONTENT' not in body(document) and 'DELTA_ORIGINAL_PAGE_THREE' not in body(document)


def test_selected_recovery_never_calls_full_page_fallback(chapter_pdf, tmp_path):
    preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    with patch.object(extractor.fallback, 'recover_page', side_effect=AssertionError('full-page recovery')):
        document = extractor.extract_pdf(chapter_pdf, tmp_path / 'recovery', fail_pages={1},
                                         extraction_scope=selection(preview, 'Beta'))
    assert document['pages'][0]['degraded']
    assert 'BETA_CHOSEN_CONTENT' in body(document)
    assert 'ALPHA_PRIVATE_CONTENT' not in body(document)
    assert 'GAMMA_CONTENT' not in body(document)


def test_shared_image_binary_is_not_exported_across_chapter_boundary(chapter_pdf, tmp_path):
    path = tmp_path / 'shared-image.pdf'
    with fitz.open(chapter_pdf) as document:
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 200), False)
        pixmap.clear_with(100)
        document[0].insert_image(fitz.Rect(400, 40, 450, 650), stream=pixmap.tobytes('png'))
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    with patch.object(fitz.Document, 'extract_image', side_effect=AssertionError('unrestricted image binary')):
        document = extractor.extract_pdf(path, tmp_path / 'shared',
                                         extraction_scope=selection(preview, 'Beta'))
    assets = document['pages'][0]['assets']
    assert assets and all(not asset['original_binary'] for asset in assets)
    assert all(asset['bbox'][1] >= preview['sections'][1]['start_anchor']['y'] for asset in assets)


def test_last_selected_page_keeps_original_page_number(chapter_pdf, tmp_path):
    preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    document = extractor.extract_pdf(chapter_pdf, tmp_path / 'last',
                                     extraction_scope=selection(preview, 'Delta'))
    assert [page['page'] for page in document['pages']] == [3]
    assert document['parts'][0]['locator']['pages'] == [3]
    assert document['summary']['original_page_count'] == 3
    assert document['summary']['all_pages_represented']
    assert (tmp_path / 'last/raw/page-0003.json').is_file()
    assert not (tmp_path / 'last/raw/page-0001.json').exists()


def test_unreliable_outline_has_no_fabricated_chapters(tmp_path):
    path = tmp_path / 'no-outline.pdf'
    with fitz.open() as document:
        for _ in range(2):
            page = document.new_page()
            page.insert_text((40, 60), 'Ordinary source prose without an authored chapter structure.')
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    assert not preview['outline_available']
    assert preview['sections'] == [] and preview['warnings']


def test_chapters_sharing_an_ambiguous_column_boundary_offer_full_document_only(tmp_path):
    path = tmp_path / 'columns.pdf'
    with fitz.open() as document:
        page = document.new_page(width=600, height=700)
        page.insert_text((40, 60), 'First chapter', fontsize=20, fontname='hebo')
        page.insert_text((40, 300), 'FIRST_COLUMN_CONTINUES', fontsize=10)
        page.insert_text((330, 300), 'Second chapter', fontsize=20, fontname='hebo')
        document.set_toc([[1, 'First chapter', 1], [1, 'Second chapter', 1]])
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    assert not preview['outline_available']
    assert preview['sections'] == [] and any('column' in warning for warning in preview['warnings'])


def test_numbered_heading_scan_is_a_text_only_outline_fallback(tmp_path):
    path = tmp_path / 'numbered.pdf'
    with fitz.open() as document:
        for number in (1, 2):
            page = document.new_page()
            page.insert_text((40, 60), f'{number} Chapter title', fontsize=22, fontname='hebo')
            page.insert_text((40, 100), 'Repeated source body text is ordinary prose.' * 3, fontsize=11)
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    assert preview['outline_available']
    assert preview['outline_source'] == 'layout_headings'
    assert [item['page_start'] for item in preview['sections']] == [1, 2]


def test_printed_toc_resolves_titles_without_guessing_front_matter_offsets(tmp_path):
    path = tmp_path / 'printed-toc.pdf'
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), 'Contents', fontsize=20)
        page.insert_text((40, 100), 'Authored First ........ 1', fontsize=11)
        page.insert_text((40, 125), 'Authored Second ........ 2', fontsize=11)
        for title in ('Authored First', 'Authored Second'):
            page = document.new_page()
            page.insert_text((40, 60), title, fontsize=22, fontname='hebo')
            page.insert_text((40, 100), 'The chapter body retains its actual physical PDF page.', fontsize=11)
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    assert preview['outline_available']
    assert preview['outline_source'] == 'printed_toc'
    assert [item['page_start'] for item in preview['sections']] == [2, 3]


def test_partial_printed_toc_match_offers_full_document_instead_of_incomplete_chapters(tmp_path):
    path = tmp_path / 'partial-printed-toc.pdf'
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), 'Contents', fontsize=20)
        for row, (title, page_number) in enumerate([('First chapter', 2), ('Missing heading', 3), ('Last chapter', 4)]):
            page.insert_text((40, 100 + row * 25), f'{title} ........ {page_number}', fontsize=11)
        for title in ('First chapter', 'An unmatched middle title', 'Last chapter'):
            page = document.new_page()
            page.insert_text((40, 60), title, fontsize=22, fontname='hebo')
            page.insert_text((40, 100), 'Substantive text belongs to this independent chapter.', fontsize=11)
        document.save(path)
    preview = preview_pdf_outline(path, source_id='SRC1')
    assert not preview['outline_available']
    assert preview['sections'] == []
    assert any('printed contents' in message and 'complete PDF' in message for message in preview['warnings'])


def test_unknown_chapter_ids_are_rejected(chapter_pdf, tmp_path):
    preview = preview_pdf_outline(chapter_pdf, source_id='SRC1')
    scope = selection(preview, 'Beta')
    scope['section_ids'] = ['not-in-original-outline']
    with pytest.raises(ValueError, match='Unknown or empty'):
        extractor.extract_pdf(chapter_pdf, tmp_path / 'unknown', extraction_scope=scope)


@pytest.mark.parametrize(('title', 'first_page', 'chosen_text'), [
    ('Delta', 3, 'DELTA_ORIGINAL_PAGE_THREE'), ('Beta', 1, 'BETA_CHOSEN_CONTENT'),
])
def test_scoped_inventory_keeps_sourceparts_and_evidence_on_original_page(chapter_pdf, tmp_path, title, first_page, chosen_text):
    settings = Settings(data_dir=tmp_path / 'data', output_dir=tmp_path / 'out')
    store = RunStore(settings.data_dir, 'run_scope_integration')
    staged = store.stage_upload('inp_1', 'chapters.pdf', chapter_pdf.read_bytes())
    target = TargetProfile(software_name='Example', target_version='1', platform='General', ui_locale='English')
    acquisition = AcquisitionService(store, settings)
    manifest = acquisition.acquire([StagedInput(input_id='inp_1', kind='uploaded_file',
        display_name='chapters.pdf', staged_path=str(staged), declared_media_type='application/pdf')], target).manifests[0]
    preview = preview_pdf_outline(chapter_pdf, source_id=manifest.source_id)
    manifest.source_metadata['extraction_scope'] = selection(preview, title)
    selected = next(section for section in preview['sections'] if section['title'] == title)
    workspace = SourceWorkspace(store, settings, acquisition, target, manifest)
    response = ToolExecutor(workspace, 'scope_test').call('pdf_inventory', {})
    assert response['ok'], response
    candidates = workspace.evidence_candidates()
    assert candidates
    assert all(first_page <= part.locator.get('page_start', part.locator.get('page')) <= selected['page_end']
               for part in candidates)
    assert workspace.pdf_document_structure.get('native_bookmarks', []) == []
    assert workspace.pdf_document_structure['authored_outline_source'] != 'native_bookmarks'
    projected = '\n'.join(workspace.part_text.get(part.part_id, '') for part in candidates)
    assert chosen_text in projected
    assert 'ALPHA_PRIVATE_CONTENT' not in projected and 'GAMMA_CONTENT' not in projected
