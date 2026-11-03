const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');

function createRenderer() {
  const list = {
    details: [], nextDetails: [], writes: 0,
    set innerHTML(value) {
      this.markup = value;
      this.details = this.nextDetails;
      this.writes++;
    },
    querySelectorAll() { return this.details; },
  };
  const context = vm.createContext({
    document: { addEventListener() {}, querySelector: () => list },
  });
  vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
  const state = vm.runInContext('appState', context);
  state.currentRunId = 'run-a';
  return {
    context, list, state,
    render(markup, details = []) {
      list.nextDetails = details;
      context.updateEvidenceTreeList(markup);
    },
  };
}

const detail = (key, open = false) => ({ dataset: { evidenceDetailKey: key }, open });

test('unchanged polling retains the actual evidence elements and their open and closed states', () => {
  const { render, list } = createRenderer();
  const root = detail('root', true);
  const leaf = detail('leaf');
  render('snapshot-a', [root, leaf]);
  root.open = false;
  leaf.open = true;
  for (let index = 0; index < 5; index++) render('snapshot-a');
  assert.equal(list.writes, 1);
  assert.equal(list.details[0], root);
  assert.equal(list.details[1], leaf);
  assert.equal(root.open, false);
  assert.equal(leaf.open, true);
});

test('updated evidence restores tree, exact-content and provenance disclosures including closed roots', () => {
  const { render, list } = createRenderer();
  render('snapshot-a', [detail('root'), detail('leaf', true), detail('material', true), detail('refs', true)]);
  const updated = [detail('root', true), detail('leaf'), detail('material'), detail('refs'), detail('new')];
  render('snapshot-b', updated);
  assert.deepEqual(updated.map(item => item.open), [false, true, true, true, false]);
  assert.equal(list.writes, 2);
});

test('temporarily filtered-out evidence keeps its disclosure state when it returns', () => {
  const { render } = createRenderer();
  render('all', [detail('source-a/leaf', true), detail('source-b/leaf')]);
  render('filtered', []);
  const returned = [detail('source-a/leaf'), detail('source-b/leaf', true)];
  render('all', returned);
  assert.deepEqual(returned.map(item => item.open), [true, false]);
});

test('switching runs resets disclosure state even when IDs and markup match', () => {
  const { render, state, list } = createRenderer();
  render('same', [detail('leaf', true)]);
  state.currentRunId = 'run-b';
  const nextRunLeaf = detail('leaf');
  render('same', [nextRunLeaf]);
  assert.equal(nextRunLeaf.open, false);
  assert.equal(list.writes, 2);
  assert.equal(state.evidenceTreeOpenState.size, 0);
});

test('source and search context separate disclosure keys while preserving escaped source content', () => {
  const { context } = createRenderer();
  const node = {
    evidence_node_id: 'N1', node_key: 'leaf', title: 'Leaf <sample>',
    content: 'Exact <source> & text', source_refs: [{ source_part_id: 'P1', locator: {} }],
  };
  const render = (sourceId, query) => context.renderEvidenceTreeNode(node, { N1: node }, new Set(['N1']), 2, query, sourceId);
  const keys = html => [...html.matchAll(/data-evidence-detail-key="([^"]+)"/g)].map(match => match[1]);
  const ordinary = render('SRC1', '');
  assert.equal(new Set(keys(ordinary)).size, 3);
  assert.notDeepEqual(keys(ordinary), keys(render('SRC2', '')));
  assert.notDeepEqual(keys(ordinary), keys(render('SRC1', 'sample')));
  assert.match(ordinary, /Leaf &lt;sample&gt;/);
  assert.match(ordinary, /Exact &lt;source&gt; &amp; text/);
  assert.doesNotMatch(ordinary, /data-evidence-detail-key="[^"]+" open/);
  assert.match(render('SRC1', 'sample'), /data-evidence-detail-key="[^"]+" open/);
});
