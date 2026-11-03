const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
function renderer() {
  const list = {
    details: [], nextDetails: [], writes: 0,
    set innerHTML(value) { this.markup = value; this.details = this.nextDetails; this.writes++; },
    querySelectorAll() { return this.details; },
  };
  const context = vm.createContext({ document: { addEventListener() {}, querySelector: () => list } });
  vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
  const state = vm.runInContext('appState', context);
  state.currentRunId = 'run-a';
  return { context, state, list, update(markup, details = []) {
    list.nextDetails = details;
    context.updateSkillsList(markup);
  } };
}
const detail = (key, open = false) => ({ dataset: { skillKey: key }, open });

test('Skills start collapsed with their title and description in the summary and all content in the body', () => {
  const { context, state, list } = renderer();
  state.runPayload = { result: { skills: [{ skill_id: 'S1', title: 'Plot <x>', description: 'Create & inspect a plot.',
    applicability: 'A session is open.', preconditions: 'Load <data>.', trajectory_ids: ['TR1'],
    procedure: [{ instruction: 'Plot the data.', citations: [] }] }] } };
  const before = JSON.stringify(state.runPayload);
  context.renderSkills();
  assert.match(list.markup, /<details class="skill-details" data-skill-key="S1">/);
  assert.match(list.markup, /<summary class="skill-summary">[\s\S]*Plot &lt;x&gt;[\s\S]*Create &amp; inspect a plot\.[\s\S]*<\/summary>/);
  assert.match(list.markup, /class="skill-details-body"[\s\S]*A session is open\.[\s\S]*Load &lt;data&gt;\.[\s\S]*Plot the data\./);
  assert.equal(JSON.stringify(state.runPayload), before);
});

test('unchanged polling keeps the original Skill element and a deliberate collapse', () => {
  const { update, list } = renderer();
  const first = detail('S1');
  update('same', [first]);
  first.open = true;
  update('same');
  assert.equal(list.details[0], first);
  assert.equal(first.open, true);
  first.open = false;
  update('same');
  assert.equal(first.open, false);
  assert.equal(list.writes, 1);
});

test('new and reordered Skills retain independent expansion choices using their IDs', () => {
  const { update } = renderer();
  update('before', [detail('S1', true), detail('S2')]);
  const changed = [detail('S2', true), detail('S3'), detail('S1')];
  update('after', changed);
  assert.deepEqual(changed.map(item => item.open), [false, false, true]);
});

test('a temporary empty result keeps expansion but switching runs resets it', () => {
  const { update, state } = renderer();
  update('before', [detail('S1', true)]);
  update('empty');
  const returned = detail('S1');
  update('before', [returned]);
  assert.equal(returned.open, true);
  state.currentRunId = 'run-b';
  const otherRun = detail('S1');
  update('before', [otherRun]);
  assert.equal(otherRun.open, false);
});

test('legacy Skills without IDs receive stable escaped keys independent of list order', () => {
  const { context, state, list } = renderer();
  const skills = [
    { title: 'Plot "<data>"', description: 'Inspect & render.', trajectory_ids: ['TR1'] },
    { title: 'Reset the view', description: 'Reset axes.' },
  ];
  state.runPayload = { result: { skills } };
  context.renderSkills();
  const keys = html => [...html.matchAll(/data-skill-key="([^"]+)"/g)].map(match => match[1]);
  const original = keys(list.markup);
  assert.equal(new Set(original).size, 2);
  assert.match(original[0], /&lt;data&gt;/);
  state.runPayload.result.skills = [...skills].reverse();
  context.renderSkills();
  assert.deepEqual(keys(list.markup), original.reverse());
});
