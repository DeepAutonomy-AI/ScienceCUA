const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');

function createRenderer() {
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, {
      textContent: '', innerHTML: '', value: '', open: false,
      showModal() { this.open = true; }, querySelectorAll() { return []; },
    });
    return nodes.get(selector);
  };
  const context = vm.createContext({ window: {}, document: {
    addEventListener() {}, querySelector: node,
  } });
  vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
  return { context, node };
}

test('new Reference descriptions and knowledge body both render with escaping', () => {
  const { context } = createRenderer();
  const html = context.renderReferenceText({
    description: 'Look up <commands>.', content: 'reset accepts <force> & preserves names.',
  });
  assert.match(html, /Look up &lt;commands&gt;\./);
  assert.match(html, /reset accepts &lt;force&gt; &amp; preserves names\./);
  assert.doesNotMatch(html, /<commands>|<force>/);
});

test('historical content displays once without mutating saved data', () => {
  const { context } = createRenderer();
  const reference = { content: 'Legacy lookup scope and content.' };
  const before = JSON.stringify(reference);
  const html = context.renderReferenceText(reference);
  assert.equal(html, '<p>Legacy lookup scope and content.</p>');
  assert.equal(JSON.stringify(reference), before);
});

test('full Reference dialog uses lookup description and keeps optional body', () => {
  const { context, node } = createRenderer();
  context.openReferenceCatalog({
    title: 'Command lookup', description: 'Consult when checking command names.',
    content: 'Commands are case sensitive.', reference_mode: 'source_catalog',
    entries: [],
  });
  assert.match(node('#reference-catalog-dialog-description').textContent, /Consult when checking command names\./);
  assert.doesNotMatch(node('#reference-catalog-dialog-description').textContent, /case sensitive/);
  assert.match(node('#reference-catalog-entry-list').innerHTML, /Commands are case sensitive\./);
});

test('historical image-keyed entries keep their text and table without images or placeholders', () => {
  const { context, node } = createRenderer();
  vm.runInContext('appState.currentRunId = "run_images"', context);
  const artifactId = 'art_' + 'a'.repeat(24);
  context.openReferenceCatalog({
    title: 'Table controls', description: 'Look up table icons.', entries: [{
      title: '[image I15]', display_title: 'Adds a row after the current row',
      content: '| [image I15] | Adds a row after the current row |', source_evidence_id: 'E1',
      image_refs: [{ artifact_id: artifactId, marker: '[image I15]', media_type: 'image/svg+xml' }],
      citations: [],
    }],
  });
  const html = node('#reference-catalog-entry-list').innerHTML;
  assert.match(html, /<h3>Adds a row after the current row<\/h3>/);
  assert.match(html, /<table class="reference-table">/);
  assert.doesNotMatch(html, /<img|\/artifacts\/|reference-image|Image unavailable/);
  assert.doesNotMatch(html, /\[image I15\]|\| Adds/);
});

test('unresolved image placeholders and image markup disappear while ordinary text stays escaped', () => {
  const { context } = createRenderer();
  const html = context.renderReferenceContent('[image I9] <img src=x onerror=alert(1)> ![icon](images/icon.svg) Keep <command>.');
  assert.match(html, /Keep &lt;command&gt;\./);
  assert.doesNotMatch(html, /<img|\[image|Image unavailable|images\/|onerror/);
});

test('Reference summary, catalog title and citation quotes are text-only for historical payloads', () => {
  const { context, node } = createRenderer();
  const reference = { title: 'Commands [image I1]', description: 'Lookup [image I2]',
    content: 'Body ![icon](images/body.png)', image_refs: [{ artifact_id: 'art_' + 'a'.repeat(24) }],
    entries: [{ title: '[image I3]', content: '| [image I3] | Insert a row |', source_evidence_id: 'E1',
      citations: [{ evidence_id: 'E1', quote: 'Insert [image I3] ![icon](images/citation.png)' }] }] };
  const before = JSON.stringify(reference);
  const summary = context.renderReferenceText(reference);
  context.openReferenceCatalog(reference);
  assert.doesNotMatch(summary + node('#reference-catalog-dialog-title').textContent +
    node('#reference-catalog-dialog-description').textContent + node('#reference-catalog-entry-list').innerHTML,
    /<img|\[image|Image unavailable|images\//);
  assert.match(node('#reference-catalog-entry-list').innerHTML, /Insert a row/);
  assert.equal(JSON.stringify(reference), before);
});

test('Reference tables keep escaped and code pipes in their own cells', () => {
  const { context } = createRenderer();
  const html = context.renderReferenceContent('| Value | Meaning |\n| --- | --- |\n| `a|b` | x\\|y |');
  assert.equal((html.match(/<td>/g) || []).length, 2);
  assert.match(html, /<th>Value<\/th>/);
  assert.match(html, /<td>`a\|b`<\/td>/);
});
