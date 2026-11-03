const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');

function createRenderer() {
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, { innerHTML: '', textContent: '', querySelectorAll: () => [] });
    return nodes.get(selector);
  };
  const context = vm.createContext({
    document: {
      addEventListener() {},
      querySelector: node,
    },
  });
  vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
  return {
    preconditions: (value) => context.renderSkillPreconditions(value),
    skills(values) {
      context.skillFixtures = values;
      vm.runInContext('appState.runPayload = { result: { skills: skillFixtures } }; renderSkills();', context);
      return node('#skills-list').innerHTML;
    },
    trajectories(values, tasks = []) {
      context.renderTrajectoryList(values, tasks);
      return node('#trajectory-list').innerHTML;
    },
    referenceEntries(value) {
      context.renderReferenceCatalogEntries(value, '');
      return node('#reference-catalog-entry-list').innerHTML;
    },
  };
}

test('current preconditions preserve line breaks and escape HTML', () => {
  const html = createRenderer().preconditions('Open the graph.\nUse <sample> & verify the input.');
  assert.match(html, /Open the graph\.<br>Use &lt;sample&gt; &amp; verify the input\./);
  assert.doesNotMatch(html, /<sample>/);
});

test('saved criterion arrays display their text without changing saved data', () => {
  const saved = [
    { criterion: 'Open the graph.', citations: [{ evidence_id: 'E1', quote: 'Open the graph.' }] },
    { criterion: 'Use <sample> & verify the input.' },
  ];
  const before = JSON.stringify(saved);
  const html = createRenderer().preconditions(saved);
  assert.match(html, /Open the graph\.<br>Use &lt;sample&gt; &amp; verify the input\./);
  assert.doesNotMatch(html, /\[object Object\]/);
  assert.equal(JSON.stringify(saved), before);
});

test('plain text arrays retain their nonempty requirements', () => {
  const html = createRenderer().preconditions(['Open the graph.', '', '  ', 'Load the input.']);
  assert.match(html, /Open the graph\.<br>Load the input\./);
});

test('empty required preconditions and missing historical values do not break rendering', () => {
  const renderer = createRenderer();
  for (const value of [undefined, null, '', '  ', [], [null, {}, 42], 42, false, {}]) {
    assert.equal(renderer.preconditions(value), '');
  }
});

test('the Skills list renders new and historical results together', () => {
  const base = {
    description: 'Render the supplied graph and inspect the output.',
    applicability: 'A graph is available.',
    procedure: [{ instruction: 'Render the graph.', citations: [] }],
    trajectory_ids: ['TR1'],
  };
  const skills = [
    { ...base, title: 'Render a graph', preconditions: 'Open the graph.' },
    { ...base, title: 'Inspect a graph', preconditions: [{ criterion: 'Load the input.' }] },
  ];
  const before = JSON.stringify(skills);
  const html = createRenderer().skills(skills);
  assert.equal((html.match(/class="skill-item"/g) || []).length, 2);
  assert.match(html, /Open the graph\./);
  assert.match(html, /Load the input\./);
  assert.doesNotMatch(html, /\[object Object\]/);
  assert.equal(JSON.stringify(skills), before);
});

test('historical Skills ignore retired fields while retaining trajectory links and citations', () => {
  const skill = {
    title: 'Render a graph', description: 'Render the current graph.',
    applicability: 'A graph is open.', preconditions: '',
    procedure: [{ instruction: 'Render the graph.', expected_observation: 'Retired Skill observation', citations: [
      { evidence_id: 'E1', quote: 'Render the graph.', rationale: 'Retired citation explanation' },
    ] }], task_ids: ['T1'], confidence: 0.97, inferred: true,
    caveats: ['Retired Skill caveat'],
    completion_criteria: [{ criterion: 'Retired completion requirement', citations: [{ evidence_id: 'E2' }] }],
    trajectory_ids: ['TR1'], reference_ids: ['R1'], reference_trigger_type: 'required',
  };
  const before = JSON.stringify(skill);
  const html = createRenderer().skills([skill]);
  assert.match(html, /1 trajectories/);
  assert.match(html, /citation-badges/);
  assert.match(html, /E1/);
  assert.doesNotMatch(html, /tasks|confidence|inferred|skill-caveats|skill-completion-criteria|Retired|E2|skill-observation/);
  assert.doesNotMatch(html, /skill-references|\bR1\b|references|reference_trigger_type/);
  assert.equal(JSON.stringify(skill), before);
});

test('Trajectory retains its steps and citations while ignoring retired metadata', () => {
  const trajectory = {
    trajectory_id: 'TR1', task_id: 'T1', goal: 'Retired duplicate description.', confidence: 0.8,
    citations: [{ evidence_id: 'E1', quote: 'Render the graph.' }],
    steps: [{ instruction: 'Render the graph.', expected_observation: 'A graph appears.',
      citations: [{ evidence_id: 'E2', quote: 'A graph appears.' }] }],
    caveats: ['Use the documented application version.'], reference_ids: ['R1'],
  };
  const before = JSON.stringify(trajectory);
  const html = createRenderer().trajectories([trajectory], [{task_id: 'T1', task: 'Render <a graph>'}]);
  assert.match(html, /Render &lt;a graph&gt;/);
  assert.doesNotMatch(html, /Retired duplicate description|80% confidence|Use the documented application version|R1/);
  assert.match(html, /A graph appears\./);
  assert.match(html, /E1/);
  assert.match(html, /E2/);
  assert.equal(JSON.stringify(trajectory), before);
});

test('Trajectory without a loaded Task shows its binding and keeps its steps', () => {
  const html = createRenderer().trajectories([{ trajectory_id: 'TR1', task_id: 'T1',
    steps: [{instruction: 'Render the graph.', citations: []}], citations: [] }]);
  assert.match(html, /<h4>T1<\/h4>/);
  assert.match(html, /Render the graph\./);
  assert.doesNotMatch(html, /undefined/);
});

test('Reference citations keep quotes and Evidence badges while ignoring retired rationale', () => {
  const reference = { entries: [{ title: 'Render', content: 'Render the current graph.',
    source_evidence_id: 'E1', citations: [{ evidence_id: 'E1', quote: 'Use <render>.',
      rationale: 'Retired Reference citation explanation' }] }] };
  const before = JSON.stringify(reference);
  const html = createRenderer().referenceEntries(reference);
  assert.match(html, /Use &lt;render&gt;\./);
  assert.match(html, /citation-badges/);
  assert.match(html, /E1/);
  assert.doesNotMatch(html, /Retired|reference-citation-rationale/);
  assert.equal(JSON.stringify(reference), before);
});
