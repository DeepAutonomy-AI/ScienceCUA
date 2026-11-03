const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function createApp() {
  const nodes = new Map();
  const fields = {};
  const requests = [];
  const node = selector => {
    assert.ok(!['#generation-coverage-current', '#generation-coverage-capacity', '#generation-coverage-locked'].includes(selector), 'Renderer queried a removed coverage detail');
    const name = selector.match(/\[name=['"]([^'"]+)['"]\]/)?.[1];
    if (name && fields[name]) return fields[name];
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', hidden: false, disabled: false, open: false, dataset: {},
      attributes: new Map(), handlers: new Map(), validity: {valid: true},
      style: {setProperty(name, value) { this[name] = value; }},
      classList: {add() {}, remove() {}, toggle() {}},
      setAttribute(name, value) { this.attributes.set(name, String(value)); },
      getAttribute(name) { return this.attributes.get(name) ?? null; }, removeAttribute() {},
      addEventListener(name, handler) { this.handlers.set(name, handler); },
      querySelector: child => node(selector + ' ' + child), contains: () => false,
    });
    return nodes.get(selector);
  };
  for (const name of ['software_name','target_version','platform','api_base_url','api_model','api_protocol','api_key','reasoning_effort']) {
    const field = node('#field-' + name); field.name = name; fields[name] = field;
  }
  fields.software_name.value = 'App'; fields.target_version.value = '1'; fields.platform.value = 'General';
  fields.api_base_url.value = 'https://example.test/v1'; fields.api_model.value = 'model';
  fields.api_key.value = 'sample-memory-only-key';
  const buttons = ['low','medium','high','ultra'].map(value => {
    const button = node('[data-coverage-level="' + value + '"]'); button.dataset.coverageLevel = value; return button;
  });
  class FormData {
    constructor() { this.values = new Map(Object.values(fields).filter(field => !field.disabled).map(field => [field.name, field.value])); }
    set(key, value) { this.values.set(key, value); } get(key) { return this.values.get(key); }
    delete(key) { this.values.delete(key); } append(key, value) { this.values.set(key, value); }
    keys() { return this.values.keys(); }
  }
  const context = vm.createContext({URL, FormData, window: {}, document: {
    addEventListener() {}, querySelector: node,
    querySelectorAll(selector) {
      if (selector === '[data-coverage-level]') return buttons;
      if (selector === '#run-form [required]') return ['software_name','target_version','platform','api_base_url','api_model'].map(name => fields[name]);
      return [];
    },
  }, fetch: async (url, options) => { requests.push({url, options}); return {ok: true, json: async () => ({run_id: 'accepted'})}; }});
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  vm.runInContext('selectRun = refreshRuns = async () => {};', context);
  context.apiFetch = context.fetch;
  const state = vm.runInContext('appState', context);
  state.user = {id: 'test-user'};
  state.sourceMode = 'discovery'; state.sourceModeChosen = true; state.setupStage = 'model';
  const submit = () => context.submitRun({preventDefault() {}, target: {}});
  return {context, state, node, buttons, requests, submit};
}

test('Medium is the default budget and the card belongs to the model setup step', () => {
  const app = createApp(); app.state.setupStage = 'target'; app.context.bindGenerationCoverage();
  assert.equal(app.node('#generation-coverage').hidden, true);
  app.state.setupStage = 'model'; app.context.renderGenerationCoverage();
  assert.equal(app.node('#generation-coverage').hidden, false);
  assert.equal(app.node('#generation-coverage-value').value, 'medium');
  assert.equal(app.node('#generation-coverage-slider').value, '1');
  assert.equal(app.node('#generation-coverage-slider').getAttribute('aria-valuetext'), 'Medium');
});

test('coverage shows one cost-and-time hint without duplicate labels or internal capacity details', () => {
  const html = fs.readFileSync(path.resolve(__dirname, '../../app/static/index.html'), 'utf8');
  const card = html.match(/<section class="form-section generation-coverage"[\s\S]*?<\/section>/)[0];
  assert.match(card, /data-i18n="Generation Coverage">Generation Coverage<\/h2>/);
  assert.equal((card.match(/<p\b/g) || []).length, 1);
  assert.match(card, /Higher coverage increases API costs and takes longer\./);
  assert.doesNotMatch(card, /<output\b|Tasks|References|coverage-description|coverage-capacity|coverage-locked|guaranteed/);
  assert.match(card, /aria-describedby="generation-coverage-help"/);
  const app = createApp(); app.state.language = 'Chinese';
  assert.equal(app.context.t('Higher coverage increases API costs and takes longer.'), '覆盖度越高，API 费用越高，耗时越长。');
});

test('label clicks and native slider input select exact allowed budgets in both languages', () => {
  const app = createApp(); app.context.bindGenerationCoverage();
  const expectations = ['low','medium','high','ultra'];
  for (let index = 0; index < expectations.length; index++) {
    const level = expectations[index];
    app.buttons[index].handlers.get('click')();
    assert.equal(app.state.generationCoverage, level);
    assert.equal(app.buttons[index].getAttribute('aria-pressed'), 'true');
    assert.equal(app.node('#generation-coverage').getAttribute('data-level'), level);
  }
  app.node('#generation-coverage-slider').handlers.get('input')({target: {value: '2'}});
  assert.equal(app.state.generationCoverage, 'high');
  app.state.language = 'Chinese'; app.context.renderGenerationCoverage();
  assert.equal(app.node('#generation-coverage-slider').getAttribute('aria-valuetext'), '高');
  app.context.setGenerationCoverage('unlimited');
  assert.equal(app.state.generationCoverage, 'high');
});

for (const level of ['low','medium','high','ultra']) {
  test(`new ${level} run sends the named preset, without client-provided numeric limits`, async () => {
    const app = createApp(); app.context.setGenerationCoverage(level); await app.submit();
    assert.equal(app.requests.length, 1);
    const form = app.requests[0].options.body;
    assert.equal(form.get('generation_coverage'), level);
    assert.equal(form.get('max_task_candidates'), undefined);
    assert.equal(form.get('max_knowledge_candidates'), undefined);
    assert.equal(form.get('reasoning_effort'), '');
  });
}

test('restored active and waiting runs keep their saved coverage and disable both controls', () => {
  const app = createApp(); app.context.bindGenerationCoverage(); app.state.currentRunId = 'existing';
  app.state.runPayload = {state: {generation_coverage: 'ultra', status: 'running', active_task: true}};
  app.context.restoreGenerationCoverage(app.state.runPayload.state); app.context.renderGenerationCoverage();
  assert.equal(app.state.generationCoverage, 'ultra');
  assert.equal(app.node('#generation-coverage-slider').disabled, true);
  assert.match(app.node('#generation-coverage-slider').getAttribute('title'), /coverage is saved/);
  assert.ok(app.buttons.every(button => button.disabled));
  app.context.setGenerationCoverage('low'); assert.equal(app.state.generationCoverage, 'ultra');
  app.state.runPayload.state = {generation_coverage: 'ultra', status: 'waiting_for_input', phase: 'awaiting_scope_selection'};
  app.context.setGenerationCoverage('high'); assert.equal(app.state.generationCoverage, 'ultra');
});

test('completed-run drafts survive refresh and an explicit T5 rerun uses the new choice', async () => {
  const app = createApp(); app.state.currentRunId = 'completed';
  app.state.runPayload = {state: {generation_coverage: 'low', status: 'completed'}};
  app.context.restoreGenerationCoverage(app.state.runPayload.state);
  app.context.setGenerationCoverage('ultra');
  app.context.restoreGenerationCoverage(app.state.runPayload.state);
  assert.equal(app.state.generationCoverage, 'ultra');
  await app.context.rerunReferenceFromT5();
  assert.equal(app.requests[0].url, '/api/runs/completed/rerun-reference');
  assert.equal(app.requests[0].options.body.get('generation_coverage'), 'ultra');
  assert.equal(app.requests[0].options.body.get('software_name'), undefined);
  assert.equal(app.state.runPayload.state.generation_coverage, 'low');
});

test('switching runs restores each choice and historical state is left untouched', () => {
  const app = createApp(); app.state.currentRunId = 'high-run';
  app.context.restoreGenerationCoverage({generation_coverage: 'high'});
  assert.equal(app.state.generationCoverage, 'high');
  const historical = {status: 'paused', target_profile: {software_name: 'App'},
    generation_limits: {max_task_candidates: 12, max_knowledge_candidates: 20}};
  const before = JSON.stringify(historical);
  app.state.currentRunId = 'old-run'; app.state.runPayload = {state: historical};
  app.context.restoreGenerationCoverage(historical); app.context.renderGenerationCoverage();
  assert.equal(app.state.generationCoverage, 'medium');
  assert.equal(app.node('#generation-coverage-slider').disabled, true);
  assert.equal(JSON.stringify(historical), before);
});
