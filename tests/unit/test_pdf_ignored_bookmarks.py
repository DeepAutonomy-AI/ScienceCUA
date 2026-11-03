"""Bookmark metadata cannot change page-derived PDF boundaries or coverage."""
from __future__ import annotations

import copy
from unittest.mock import patch

import pymupdf as fitz
import pytest

from app.adaptive_source import SourceContext, digest, write_json
from app.pdf_sourcepart import extractor
from app.pdf_sourcepart.scope import preview_pdf_outline
from app.source_tools.pdf_structure import _pdf_bookmark_outline, _pdf_heading_candidates


TITLES = ['Getting Started', 'Display Settings', 'Moving Around',
          'Choosing Objects', 'Adjusting Time', 'Saving Views']


def manual(path, bookmarks):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 50), 'Contents', fontsize=20)
        for index, title in enumerate(TITLES):
            y = 90 + 25 * index
            page.insert_text((40, y), f'{title} ........ {index + 2}', fontsize=11)
        for index, title in enumerate(TITLES):
            page = document.new_page()
            page.insert_text((40, 60), title, fontsize=18, fontname='hebo')
            page.insert_text((40, 100), f'BODY_{index} describes an independent operation and its controls.', fontsize=11)
        page = document[0]
        for index in range(len(TITLES)):
            y = 90 + 25 * index
            page.insert_link({'kind': fitz.LINK_GOTO, 'from': fitz.Rect(38, y - 12, 420, y + 3),
                              'page': index + 1, 'to': fitz.Point(40, 40)})
        document.set_toc(bookmarks)
        document.save(path)


@pytest.mark.parametrize('bookmarks', [
    [], [[1, 'Moving Around', 4]],
    [[1, 'Metadata wrapper', 1], *[[2, title, index + 2] for index, title in enumerate(TITLES)]],
])
def test_bookmark_variants_have_identical_printed_boundaries_and_exact_body_coverage(tmp_path, bookmarks):
    source = tmp_path / 'manual.pdf'
    manual(source, bookmarks)
    with patch.object(fitz.Document, 'get_toc', side_effect=AssertionError('bookmark access')) as get_toc:
        document = extractor.extract_pdf(source, tmp_path / 'parsed', source_id='SRC1')
        preview = preview_pdf_outline(source, source_id='SRC1')
    get_toc.assert_not_called()
    assert document['outline'] == []
    assert len(document['printed_toc']) == len(TITLES)
    assert preview['outline_source'] == 'printed_toc'
    assert [item['title'] for item in preview['sections']] == TITLES
    assert [item['page_start'] for item in preview['sections']] == list(range(2, 8))
    headings = [block for page in document['pages'] if page['page'] > 1
                for block in page['blocks'] if block['type'] == 'heading']
    assert [item['title'] for item in headings] == TITLES
    assert all(item['level'] == 1 and item['heading_method'] == 'printed_toc_link' for item in headings)
    assert document['summary']['structured_line_coverage']
    projected = '\n'.join(part['content'] for part in document['parts'])
    for index in range(len(TITLES)):
        assert projected.count(f'BODY_{index}') == 1
    assert (tmp_path / 'parsed/raw/source.pdf').read_bytes() == source.read_bytes()


def test_legacy_native_hints_cannot_override_printed_titles_or_suppress_layout_headings():
    def block(part_id, text, size):
        return {'part_id': part_id, 'text': text, 'font_size': size, 'max_font_size': size,
                'bbox': [40, 200, 400, 230], 'page_height': 800, 'line_count': 1}

    contexts = [{'page': 3, 'blocks': [block('first', 'Getting Started', 18),
                                     block('second', 'Additional Controls', 18)], 'page_height': 800}]
    printed = [{'node_id': 'printed-first', 'title': 'Getting Started', 'level': 1,
                'page_start': 3, 'source': 'printed_toc', 'parent_node_id': None}]
    hostile = [{'node_id': 'native-only', 'title': 'Additional Controls', 'level': 5,
                'page_start': 3, 'source': 'native_bookmark', 'parent_node_id': 'wrong-parent'}]
    baseline = _pdf_heading_candidates(copy.deepcopy(contexts), body_font_size=11,
                                       printed_toc_entries=printed)
    actual = _pdf_heading_candidates(copy.deepcopy(contexts), body_font_size=11,
                                     bookmark_outline=hostile, printed_toc_entries=printed)
    assert actual == baseline
    assert {item['title'] for item in actual.values()} == {'Getting Started', 'Additional Controls'}
    assert {item['source'] for item in actual.values()} == {'printed_toc', 'layout_heading'}
    assert _pdf_bookmark_outline([[1, 'Metadata title', 3]], 8) == []


def test_legacy_extractor_outline_argument_cannot_suppress_page_typography(tmp_path):
    source = tmp_path / 'manual.pdf'
    manual(source, [])
    with fitz.open(source) as document:
        snapshots = [extractor.raw_page(page, index + 1) for index, page in enumerate(document)]
    expected = extractor.document_structure(copy.deepcopy(snapshots))
    actual = extractor.document_structure(copy.deepcopy(snapshots), [[1, 'Moving Around', 4]])
    assert actual == expected


def test_adaptive_legacy_outline_validates_resource_without_reading_bookmarks(tmp_path):
    source = tmp_path / 'manual.pdf'
    manual(source, [[1, 'Moving Around', 4]])
    source_hash = digest(source.read_bytes())
    assignment = tmp_path / 'assignment.json'
    write_json(assignment, {'root_sha256': source_hash})
    write_json(tmp_path / 'resources.json', {'resources': [
        {'resource_key': 'r1', 'path': source.name, 'sha256': source_hash, 'modality': 'pdf'},
    ]})
    context = SourceContext(assignment)
    with patch.object(fitz.Document, 'get_toc', side_effect=AssertionError('adaptive bookmark access')) as get_toc:
        assert context.pdf_outline() == []
        assert any('BODY_0' in unit['text'] for unit in context.units() if unit['kind'] == 'text')
    get_toc.assert_not_called()
    source.write_bytes(b'%PDF-modified')
    with pytest.raises(ValueError, match='Raw source changed'):
        context.pdf_outline()
    with pytest.raises(StopIteration):
        context.pdf_outline('missing-resource')
