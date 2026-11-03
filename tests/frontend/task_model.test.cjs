const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function render(tasks) {
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, { innerHTML: '', textContent: '' });
    return nodes.get(selector);
  };
  const context = vm.createContext({ document: { addEventListener() {}, querySelector: node } });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  context.renderTaskList(tasks);
  return node('#task-list').innerHTML;
}

test('Task displays task text, chapter and citations without retired metadata', () => {
  const html = render([{ task: 'Render <graph>', category: 'Display & rendering',
    citations: [{ evidence_id: 'E1', quote: 'Render the graph.' }],
    objective: 'Retired objective', confidence: 0.9, caveats: ['Retired note'] }]);
  assert.match(html, /Render &lt;graph&gt;/);
  assert.match(html, /Display &amp; rendering/);
  assert.match(html, /E1/);
  assert.doesNotMatch(html, /Retired|90%|<p>/);
});

test('historical Task titles remain readable', () => {
  assert.match(render([{ title: 'Saved historical task', category: 'Display', citations: [] }]), /Saved historical task/);
});

test('empty Task list renders the existing empty state', () => {
  assert.match(render([]), /No submitted artifacts yet/);
});
