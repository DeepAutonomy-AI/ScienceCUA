const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function renderer() {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      textContent: '', innerHTML: '', value: '', open: false,
      showModal() { this.open = true; }, querySelectorAll() { return []; },
    });
    return nodes.get(selector);
  };
  const context = vm.createContext({window: {}, document: {
    addEventListener() {}, querySelector: node,
  }});
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  return {context, node};
}

test('presentation renders readable source headings, table columns, code and lists safely', () => {
  const {context} = renderer();
  const text = [
    '## Operators',
    '| Operator | Alias | Precedence | Meaning |',
    '| --- | --- | --- | --- |',
    '| `**` | `^` | 2 | power, like pwr |',
    '| `a|b` | x\\|y | 3 | <script>alert(1)</script> |',
    '', '- Use `sqrt(x)`.', '- Keep values.',
    '', '1. First', '2. Second',
    '', '```text', '  <img src=x onerror=alert(1)>', '![literal](sample.png)', '[image I1]', '```',
  ].join('\n');
  const html = context.renderReferencePresentation(text);
  assert.match(html, /<h3>Operators<\/h3>/);
  assert.equal((html.match(/<th>/g) || []).length, 4);
  assert.equal((html.match(/<td>/g) || []).length, 8);
  assert.match(html, /<td><code>\*\*<\/code><\/td><td><code>\^<\/code><\/td><td>2<\/td>/);
  assert.match(html, /<code>a\|b<\/code>/);
  assert.match(html, /<td>x\|y<\/td>/);
  assert.match(html, /<ul><li>Use <code>sqrt\(x\)<\/code>\.<\/li>/);
  assert.match(html, /<ol><li>First<\/li><li>Second<\/li><\/ol>/);
  assert.match(html, /<pre><code>  &lt;img/);
  assert.match(html, /!\[literal\]\(sample\.png\)/);
  assert.match(html, /\[image I1\]/);
  assert.doesNotMatch(html, /<script|<img|onerror="/);
});

test('formatted catalog opens with presentation and retains source fragments in closed details', () => {
  const {context, node} = renderer();
  const reference = {
    reference_id: 'R1', title: 'Operators', description: 'Lookup expressions.',
    reference_mode: 'source_catalog', injection_policy: 'on_demand',
    content: '', entries: [{entry_id: 'RE1', title: '2', content: '2\npower, like pwr', source_evidence_id: 'E1', citations: []}],
    presentation: {mode: 'agent_formatted', content: '## Operators\n| Operator | Precedence |\n| --- | --- |\n| ** | 2 |'},
  };
  const original = JSON.stringify(reference);
  context.openReferenceCatalog(reference);
  const html = node('#reference-catalog-entry-list').innerHTML;
  assert.match(html, /reference-formatted-content/);
  assert.match(html, /<th>Operator<\/th>/);
  assert.match(html, /<details class="reference-source-details"><summary>View source-preserved entries<\/summary>/);
  assert.ok(html.indexOf('reference-formatted-content') < html.indexOf('reference-source-details'));
  assert.match(html, /<h3>2<\/h3>/);
  assert.doesNotMatch(node('#reference-catalog-dialog-description').textContent, /on_demand|complete entries/);
  assert.equal(JSON.stringify(reference), original);
  context.renderReferenceCatalogEntries(reference, 'missing');
  assert.match(node('#reference-catalog-entry-list').innerHTML, /No matching/);
  assert.doesNotMatch(node('#reference-catalog-entry-list').innerHTML, /reference-formatted-content/);
});

test('formatted cards show the useful title and full-reference action without fragment previews', () => {
  const {context, node} = renderer();
  context.renderReferenceListWithDetails([{
    reference_id: 'R1', reference_mode: 'source_catalog', title: 'Operators', description: 'Lookup expressions.',
    entries: [{title: 'BROKEN-FRAGMENT', content: 'raw'}],
    presentation: {mode: 'agent_formatted', content: '## Operators\n| A | B |'},
  }]);
  const html = node('#reference-list').innerHTML;
  assert.match(html, /Formatted reference/);
  assert.match(html, /View Full Reference/);
  assert.doesNotMatch(html, /BROKEN-FRAGMENT|on_demand/);
});

test('pipe operators and raw HTML stay readable instead of becoming tables or executable nodes', () => {
  const {context} = renderer();
  const html = context.renderReferencePresentation('||\n| operand |\n<script>alert(1)</script>');
  assert.match(html, /<p>\|\|<\/p>/);
  assert.match(html, /<p>\| operand \|<\/p>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<table|<script/);
});

test('source table encoding round-trips backslashes, pipes, backticks, entities and cell newlines', () => {
  const {context} = renderer();
  // These cells use the host's _cell encoding: source slash/pipe/backtick
  // escapes, source HTML entities, and only a host-added <br> for newlines.
  const text = [
    '| Operator | Meaning |', '| --- | --- |',
    '| \\\\ | integer divide |',
    '| \\|\\| | boolean or |',
    '| a\\\\\\|b | slash then pipe |',
    '| \\`x\\` | literal backticks |',
    '| &lt;br&gt; | first<br>second |',
    '| &amp;lt; | literal entity |',
    '| &lt;img src=x onerror=alert(1)&gt; | source HTML |',
  ].join('\n');
  const html = context.renderReferencePresentation(text);
  assert.equal((html.match(/<td>/g) || []).length, 14);
  assert.match(html, /<td>\\<\/td><td>integer divide<\/td>/);
  assert.match(html, /<td>\|\|<\/td><td>boolean or<\/td>/);
  assert.match(html, /<td>a\\\|b<\/td>/);
  assert.match(html, /<td>`x`<\/td>/);
  assert.match(html, /<td>&lt;br&gt;<\/td><td>first<br>second<\/td>/);
  assert.match(html, /<td>&amp;lt;<\/td>/);
  assert.doesNotMatch(html, /<img|<script/);
});

test('mixed nested source lists preserve parent-child structure and explicit ordered numbering', () => {
  const {context} = renderer();
  const text = [
    '- Outer A',
    '  3. Ordered child A',
    '    - Leaf A',
    '    - Leaf B',
    '  7. Ordered child B',
    '  8. Ordered child C',
    '- Outer B',
    '  - Unordered child',
    '    0. Zero',
    '    1. One',
    '- Outer C',
  ].join('\n');
  const html = context.renderReferencePresentation(text);
  assert.equal(html, '<ul><li>Outer A<ol start="3"><li>Ordered child A<ul><li>Leaf A</li><li>Leaf B</li></ul></li>'
    + '<li value="7">Ordered child B</li><li>Ordered child C</li></ol></li>'
    + '<li>Outer B<ul><li>Unordered child<ol start="0"><li>Zero</li><li>One</li></ol></li></ul></li>'
    + '<li>Outer C</li></ul>');
});

test('source list type changes remain siblings and arbitrary integer ordinals stay safe', () => {
  const {context} = renderer();
  const html = context.renderReferencePresentation([
    '4. Parent', '  - Child <img src=x>', '  2. Other child', '5. Next parent',
    '- Unordered sibling', '-3. Negative ordinal', '0. Reset ordinal',
  ].join('\n'));
  assert.equal(html, '<ol start="4"><li>Parent<ul><li>Child &lt;img src=x&gt;</li></ul>'
    + '<ol start="2"><li>Other child</li></ol></li><li>Next parent</li></ol>'
    + '<ul><li>Unordered sibling</li></ul><ol start="-3"><li>Negative ordinal</li><li value="0">Reset ordinal</li></ol>');
  assert.doesNotMatch(html, /<img/);
});
