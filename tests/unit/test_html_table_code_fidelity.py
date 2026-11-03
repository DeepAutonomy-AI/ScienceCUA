"""Final SourcePart table serialization must keep authored executable examples."""
import html

import pytest

from app.html_sourcepart.extractor import Extractor, all_blocks, fenced_code, verify_markdown_code_integrity


def extracted(markup, *, max_chars=10_000):
    source = '<!doctype html><html><head><title>Manual</title></head><body><main>' + markup + '</main></body></html>'
    document = Extractor('https://example.test/manual/').extract(source, max_chars=max_chars)
    assert document['extraction']['mode'] == 'structured', document['extraction']
    return document, '\n\n'.join(part['content'] for part in document['parts'])


def code_blocks(document):
    return [block for section in document['sections'] for block in all_blocks(section['blocks'])
            if block['type'] in {'code', 'command'}]


def test_ordinary_simple_table_and_inline_operators_keep_their_existing_format():
    _, text = extracted('<table><tr><th>Operator</th><th>Description</th></tr>'
        '<tr><td><code>a | b</code></td><td>First line<br>Second line</td></tr></table>')
    assert text == '| Operator | Description |\n| --- | --- |\n| `a \\| b` | First line<br>Second line |'


@pytest.mark.parametrize('header', [True, False])
def test_multiline_code_in_table_remains_a_fenced_copyable_block(header):
    code = '  G4Box("world",\n\twidth,\n\theight);\n\nif (a | b) {\n    run();\n}\n'
    heading = '<tr><th>Source</th><th>Explanation</th></tr>' if header else ''
    document, text = extracted('<table>' + heading + '<tr><td><pre><code class="language-cpp">'
        + html.escape(code) + '</code></pre></td><td>Build a box.</td></tr></table>')
    block = code_blocks(document)[0]
    assert block['language'] == 'cpp'
    assert block['runs'][0]['text'] == code
    assert fenced_code(code, 'cpp') in text
    assert '[Table — cell positions and spans preserved]' in text
    assert 'Column 1; rowspan=1; colspan=1:\n\n```cpp\n' in text
    assert ' / ' not in text and '<br>' not in text and '\\|' not in text


def test_code_in_spanning_cell_preserves_row_and_column_relationships():
    code = 'for value in values:\n    print(value)\n'
    _, text = extracted('<table><tr><td rowspan="2"><pre>' + html.escape(code)
        + '</pre></td><td>First</td></tr><tr><td>Second</td></tr></table>')
    assert 'Row 1:' in text and 'Column 1; rowspan=2; colspan=1:' in text
    assert 'Row 2:\n\nColumn 2; rowspan=1; colspan=1:' in text
    assert fenced_code(code) in text
    assert text.index('First') < text.index('Row 2:') < text.index('Second')


def test_nested_table_with_code_is_not_flattened_by_its_outer_cell():
    code = '  command --first 1 ' + '\\' + '\n    --second "a | b"\n'
    _, text = extracted('<table><tr><th>Example</th></tr><tr><td><p>Nested example:</p>'
        '<table><tr><td><pre>' + html.escape(code) + '</pre></td><td>Explanation</td></tr></table>'
        '<p>After the example.</p></td></tr></table>')
    assert text.count('[Table — cell positions and spans preserved]') == 2
    assert fenced_code(code) in text
    assert text.index('Nested example:') < text.index(fenced_code(code)) < text.index('After the example.')


def test_table_code_with_embedded_markdown_fences_uses_a_longer_safe_fence():
    code = 'print("``` literal fence")\n    print("```` another fence")\n'
    _, text = extracted('<table><tr><td><pre>' + html.escape(code) + '</pre></td></tr></table>')
    assert '`````\n' + code + '\n`````' in text


def test_multi_paragraph_and_list_cells_retain_block_boundaries():
    _, text = extracted('<table><tr><th>Steps</th></tr><tr><td><p>Before.</p><p>After.</p>'
        '<ul><li>Select input</li><li>Run analysis</li></ul></td></tr></table>')
    assert 'Before.\n\nAfter.\n\n- Select input\n- Run analysis' in text
    assert ' / ' not in text and '<br>' not in text


def test_code_table_is_not_split_mid_cell_by_sourcepart_soft_limit():
    code = ''.join(f'    result[{index}] = compute({index})\n' for index in range(25))
    document, text = extracted('<h1>Example</h1><table><tr><td><pre>' + html.escape(code)
        + '</pre></td></tr></table>', max_chars=100)
    assert fenced_code(code) in text
    matching = [part for part in document['parts'] if code in part['content']]
    assert len(matching) == 1 and matching[0]['metadata']['soft_limit_exceeded']


def test_language_is_only_taken_from_unambiguous_author_declarations():
    document, text = extracted('<table><tr><td><pre class="language-python"><code class="language-js">'
        'print("not a language detector")\n</code></pre></td></tr></table>')
    assert code_blocks(document)[0]['language'] is None
    assert '```\nprint(' in text


def test_complex_non_code_table_keeps_simple_cell_text_compatibility():
    _, text = extracted('<table><tr><td rowspan="2">Alpha</td><td>Beta</td></tr>'
        '<tr><td>Gamma</td></tr></table>')
    assert text == ('[Table — cell positions and spans preserved]\nRow 1:\n'
        '  Column 1; rowspan=2; colspan=1: Alpha\n  Column 2; rowspan=1; colspan=1: Beta\n'
        'Row 2:\n  Column 2; rowspan=1; colspan=1: Gamma')


@pytest.mark.parametrize('wrapping', [
    '<table><tr><td>{code}</td></tr></table>',
    '<table><tr><td><ul><li>{code}</li></ul></td></tr></table>',
    '<ul><li><p>Example:</p><table><tr><td>{code}</td></tr></table></li></ul>',
    '<ol start="12"><li><ul><li><blockquote>{code}</blockquote></li></ul></li></ol>',
])
def test_final_code_checker_accounts_for_tables_lists_and_quote_context(wrapping):
    code = '  if (a | b) {\n\tprint("``` literal");\n  }\n'
    document, _ = extracted(wrapping.format(code='<pre class="language-cpp">' + html.escape(code) + '</pre>'))
    for part in document['parts']:
        check = verify_markdown_code_integrity(part['metadata']['blocks'], part['content'])
        assert check['status'] == 'passed', check
    assert sum(verify_markdown_code_integrity(p['metadata']['blocks'], p['content'])['code_blocks_checked']
               for p in document['parts']) == 1


@pytest.mark.parametrize('damage', ['flatten', 'indent', 'pipe', 'language', 'fence'])
def test_final_code_checker_rejects_serialization_damage(damage):
    code = '    run(a | b);\n    finish();\n'
    document, _ = extracted('<table><tr><td><pre class="language-cpp">' + html.escape(code) + '</pre></td></tr></table>')
    part = next(p for p in document['parts'] if code in p['content'])
    text = part['content']
    if damage == 'flatten':
        text = text.replace(code, code.replace('\n', ' / '))
    elif damage == 'indent':
        text = text.replace('    run', 'run')
    elif damage == 'pipe':
        text = text.replace('a | b', 'a \\| b')
    elif damage == 'language':
        text = text.replace('```cpp', '```python')
    else:
        text = text.rsplit('```', 1)[0]
    check = verify_markdown_code_integrity(part['metadata']['blocks'], text)
    assert check['status'] == 'failed' and check['error_count'] == 1


def test_final_code_checker_reports_zero_when_there_is_no_authored_code():
    document, _ = extracted('<p>Read this documentation.</p>')
    part = document['parts'][0]
    check = verify_markdown_code_integrity(part['metadata']['blocks'], part['content'])
    assert check['status'] == 'passed' and check['code_blocks_expected'] == check['code_blocks_checked'] == 0


def test_final_code_checker_does_not_skip_real_opening_after_unrelated_closing_fence():
    # Fallback/authored text may itself contain a fenced snippet. Its language
    # opener differs from the typed code opener, while its closing looks equal.
    # Reading that closing as an opener must not skip the actual typed opener.
    from app.html_sourcepart.extractor import markdown_blocks

    blocks = [{'type': 'fallback_text', 'runs': [{'text': '```python\nunrelated example\n```'}]},
              {'type': 'code', 'runs': [{'type': 'text', 'text': '  run(a | b)\n'}]}]
    check = verify_markdown_code_integrity(blocks, markdown_blocks(blocks))
    assert check['status'] == 'passed' and check['code_blocks_checked'] == 1


def test_final_code_checker_keeps_searching_after_different_container_indent():
    from app.html_sourcepart.extractor import markdown_blocks

    blocks = [{'type': 'fallback_text', 'runs': [{'text': '```python\nunrelated example\n```'}]},
              {'type': 'list', 'ordered': False, 'start': 1, 'items': [
                  {'type': 'list_item', 'children': [
                      {'type': 'paragraph', 'runs': [{'type': 'text', 'text': 'Example:'}]},
                      {'type': 'code', 'runs': [{'type': 'text', 'text': '  run(a | b)\n'}]},
                  ]}]}]
    check = verify_markdown_code_integrity(blocks, markdown_blocks(blocks))
    assert check['status'] == 'passed' and check['code_blocks_checked'] == 1
