const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function createApp() {
  const nodes = new Map();
  let focused;
  let layoutReads = 0;
  function element(name = '') {
    const classes = new Set();
    const attributes = new Map();
    const result = {
      name, value: '', textContent: '', open: false, disabled: false, hidden: false, hints: [],
      showModal() { this.open = true; }, close() { this.open = false; },
      validity: { valid: true },
      classList: { add: (...items) => items.forEach(item => classes.add(item)),
        remove: (...items) => items.forEach(item => classes.delete(item)), contains: item => classes.has(item) },
      setAttribute: (key, value) => attributes.set(key, String(value)),
      getAttribute: key => attributes.get(key) ?? null,
      removeAttribute: key => attributes.delete(key),
      closest: () => result.wrapper || null,
      contains: field => field.name.startsWith('api_'),
      scrollIntoView(options) { this.scrollOptions = options; },
      focus() { focused = this; },
      insertAdjacentElement(position, hint) { this.hints.push(hint); hint.parent = this; },
      appendChild(hint) { this.hints.push(hint); hint.parent = this; },
      remove() { this.removed = true; },
      querySelector: selector => node(selector), querySelectorAll: () => [],
    };
    Object.defineProperty(result, 'offsetWidth', { get() { layoutReads++; return 100; } });
    return result;
  }
  const fields = Object.fromEntries(['software_name', 'target_version', 'platform', 'api_base_url', 'api_model', 'api_key']
    .map(name => { const field = element(name); field.wrapper = element(); return [name, field]; }));
  Object.assign(fields.software_name, { value: 'Scientific App' });
  Object.assign(fields.target_version, { value: '1.0' });
  Object.assign(fields.platform, { value: 'General' });
  Object.assign(fields.api_base_url, { value: 'https://example.test/v1' });
  Object.assign(fields.api_model, { value: 'test-model' });
  const required = Object.values(fields).filter(field => field.name !== 'api_key');
  function node(selector) {
    const name = selector.match(/\[name=['"]([^'"]+)['"]\]/)?.[1];
    if (fields[name]) return fields[name];
    if (!nodes.has(selector)) {
      const item = element();
      item.id = selector.startsWith('#') ? selector.slice(1) : '';
      nodes.set(selector, item);
    }
    return nodes.get(selector);
  }
  class FormData {
    constructor() { this.values = new Map(Object.values(fields).map(field => [field.name, field.value])); }
    get(key) { return this.values.get(key); }
    set(key, value) { this.values.set(key, value); }
    delete(key) { this.values.delete(key); }
    append(key, value) { this.values.set(key, value); }
  }
  const requests = [];
  const context = vm.createContext({ URL, FormData, window: { matchMedia: () => ({ matches: false }) },
    document: { addEventListener() {}, querySelector: node,
      querySelectorAll: selector => selector === '#run-form [required]' ? required : [], createElement: () => element() },
    fetch: async (url, options) => { requests.push({ url, options });
      return { ok: true, json: async () => ({ run_id: 'accepted' }) }; },
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  vm.runInContext('selectRun = refreshRuns = async () => {}; showPipelineError = () => { throw new Error("Unexpected error modal"); };', context);
  const state = vm.runInContext('appState', context);
  state.sourceMode = 'discovery';
  state.setupStage = 'model';
  state.sourceModeChosen = true;
  const submit = () => context.submitRun({ preventDefault() {}, target: {} });
  return { context, state, node, fields, requests, submit, focused: () => focused, layoutReads: () => layoutReads };
}

test('Generate highlights all missing required fields without submitting and focuses the first one', async () => {
  const app = createApp();
  app.fields.software_name.value = '  ';
  app.fields.api_model.value = '';
  await app.submit();
  assert.equal(app.requests.length, 0);
  for (const name of ['software_name', 'api_model']) {
    assert.equal(app.fields[name].getAttribute('aria-invalid'), 'true');
    assert.equal(app.fields[name].wrapper.classList.contains('validation-attention'), true);
    assert.ok(app.fields[name].getAttribute('aria-describedby'));
    assert.match(app.state.formIssues.get(app.fields[name]).hint.textContent, /Please fill in/);
  }
  assert.equal(app.focused(), app.fields.software_name);
  assert.equal(app.node('#model-configuration').open, true);
  assert.match(app.node('#form-error').textContent, /highlighted required fields/);
});

test('missing model configuration expands the collapsed section before focus and restarts attention on another click', async () => {
  const app = createApp();
  app.fields.api_model.value = '';
  await app.submit();
  assert.equal(app.node('#model-configuration').open, true);
  assert.equal(app.focused(), app.fields.api_model);
  const firstHint = app.state.formIssues.get(app.fields.api_model).hint;
  await app.submit();
  assert.equal(firstHint.removed, true);
  assert.equal(app.layoutReads(), 2);
  assert.equal(app.state.formIssues.size, 1);
  assert.equal(app.node('#model-configuration-state').textContent, 'Needs attention');
});

test('a server missing-Key rejection points directly to the Key without changing the current run or showing a generic modal', async () => {
  const app = createApp();
  app.state.language = 'Chinese';
  app.state.currentRunId = 'existing-run';
  app.context.fetch = async (url, options) => { app.requests.push({ url, options });
    return { ok: false, json: async () => ({ detail: 'API Agent requires both API Key and model name' }) }; };
  await app.submit();
  assert.equal(app.requests.length, 1);
  assert.equal(app.focused(), app.fields.api_key);
  assert.equal(app.fields.api_key.getAttribute('aria-invalid'), 'true');
  assert.equal(app.state.formIssues.get(app.fields.api_key).hint.textContent, '请先填写 API Key，再点击生成。');
  assert.equal(app.node('#model-configuration').open, true);
  assert.equal(app.node('#run-button').disabled, false);
  assert.equal(app.state.currentRunId, 'existing-run');
});

test('a blank form Key is accepted when the backend can supply the credential, and Automatic protocol stays optional', async () => {
  const app = createApp();
  await app.submit();
  assert.equal(app.requests.length, 1);
  assert.equal(app.requests[0].options.body.get('api_key'), '');
  assert.equal(app.state.formIssues.size, 0);
});

test('correcting a field clears the highlight and hint while preserving its existing help description', () => {
  const app = createApp();
  app.fields.api_key.setAttribute('aria-describedby', 'key-help');
  app.context.showRunFormIssues([{ field: app.fields.api_key, message: 'Missing Key' }]);
  const hint = app.state.formIssues.get(app.fields.api_key).hint;
  app.context.clearRunFormIssue(app.fields.api_key);
  assert.equal(hint.removed, true);
  assert.equal(app.fields.api_key.getAttribute('aria-describedby'), 'key-help');
  assert.equal(app.fields.api_key.getAttribute('aria-invalid'), null);
  assert.equal(app.fields.api_key.wrapper.classList.contains('validation-attention'), false);
  assert.equal(app.node('#form-error').textContent, '');
  assert.equal(app.node('#model-configuration-state').textContent, 'Configured');
});

test('manual mode highlights missing material while automatic discovery does not require it', async () => {
  const app = createApp();
  app.state.sourceMode = 'manual';
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.focused(), app.node('#dropzone'));
  assert.match(app.node('#form-error').textContent, /exactly one PDF/);
});

test('invalid Base URLs get actionable feedback before a request, without exposing their contents', async () => {
  const app = createApp();
  app.fields.api_base_url.value = 'https://user:secret@example.test/v1';
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.focused(), app.fields.api_base_url);
  assert.match(app.node('#form-error').textContent, /valid Base URL/);
  assert.doesNotMatch(app.node('#form-error').textContent, /secret/);
});

test('a missing manual URL places the hint below the whole field instead of inside its flex input row', async () => {
  const app = createApp();
  app.state.sourceMode = 'manual';
  app.state.manualSourceKind = 'url';
  const field = app.node('#url-list .url-value');
  field.wrapper = app.node('.source-url-field');
  await app.submit();
  assert.equal(app.focused(), field);
  assert.equal(field.wrapper.hints.length, 1);
  assert.equal(field.hints.length, 0);
});

test('reduced-motion settings retain the warning and focus without smooth scrolling', async () => {
  const app = createApp();
  app.context.window.matchMedia = () => ({ matches: true });
  app.fields.api_model.value = '';
  await app.submit();
  assert.equal(app.fields.api_model.scrollOptions.behavior, 'auto');
  assert.equal(app.fields.api_model.getAttribute('aria-invalid'), 'true');
  const css = fs.readFileSync(path.resolve(__dirname, '../../app/static/styles.css'), 'utf8');
  assert.match(css, /prefers-reduced-motion: reduce[^}]+#run-form \.validation-attention \{ animation: none;/);
  const html = fs.readFileSync(path.resolve(__dirname, '../../app/static/index.html'), 'utf8');
  assert.match(html, /<form id="run-form" novalidate>/);
});

test('entry starts at Target Environment without opening the material dialog', () => {
  const app = createApp();
  app.state.sourceModeChosen = false;
  app.fields.software_name.value = '';
  app.context.initializeSetupFlow();
  assert.equal(app.state.setupStage, 'target');
  assert.equal(app.node('#source-mode-dialog').open, false);
  assert.equal(app.node('#sources-configuration').hidden, true);
  assert.equal(app.node('#model-configuration').hidden, true);
  assert.equal(app.node('#run-button').hidden, true);
  assert.equal(app.focused(), app.fields.software_name);
});

test('Next validates only Target Environment and Enter advances without submitting a run', async () => {
  const app = createApp();
  app.state.setupStage = 'target';
  app.state.sourceModeChosen = false;
  app.fields.platform.value = '';
  app.fields.api_model.value = '';
  await app.submit();
  assert.equal(app.state.formIssues.size, 1);
  assert.equal(app.focused(), app.fields.platform);
  assert.equal(app.node('#source-mode-dialog').open, false);
  app.fields.platform.value = 'Linux';
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.state.setupStage, 'sources');
  assert.equal(app.node('#source-mode-dialog').open, true);
  assert.equal(app.node('#model-configuration').hidden, true);
  assert.equal(app.state.formIssues.size, 0);
});

test('manual branch requires material before revealing model settings, then focuses the first missing setting', async () => {
  const app = createApp();
  app.state.setupStage = 'sources';
  app.context.chooseSourceMode('manual');
  assert.equal(app.node('#source-mode-dialog').open, false);
  assert.equal(app.node('#continue-to-model').hidden, false);
  assert.equal(app.node('#model-configuration').hidden, true);
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.focused(), app.node('#dropzone'));
  app.state.files = [{ file: { name: 'manual.pdf' } }];
  app.fields.api_model.value = '';
  await app.submit();
  assert.equal(app.requests.length, 0);
  assert.equal(app.state.setupStage, 'model');
  assert.equal(app.node('#model-configuration').hidden, false);
  assert.equal(app.node('#model-configuration').open, true);
  assert.equal(app.focused(), app.fields.api_model);
  assert.equal(app.node('#run-button').hidden, false);
});

test('manual URLs are validated before the model step and corrected links can proceed', () => {
  const app = createApp();
  app.context.chooseSourceMode('manual');
  app.state.manualSourceKind = 'url';
  app.state.urls = [{ url: 'invalid manual' }];
  app.context.continueToModelConfiguration();
  assert.equal(app.state.setupStage, 'sources');
  assert.match(app.node('#form-error').textContent, /valid manual URL/);
  app.state.urls[0].url = 'https://example.test/manual/';
  app.context.continueToModelConfiguration();
  assert.equal(app.state.setupStage, 'model');
});

test('discovery skips uploading and returning to Target Environment preserves all drafts', () => {
  const app = createApp();
  app.fields.api_key.value = 'in-memory-only';
  const file = { file: { name: 'draft.pdf' } };
  app.state.files = [file];
  app.context.chooseSourceMode('discovery');
  assert.equal(app.state.setupStage, 'model');
  assert.equal(app.node('#manual-source-inputs').hidden, true);
  assert.equal(app.node('#model-configuration').open, true);
  app.context.backToTargetEnvironment();
  assert.equal(app.state.setupStage, 'target');
  assert.equal(app.node('#sources-configuration').hidden, true);
  assert.equal(app.fields.software_name.value, 'Scientific App');
  assert.equal(app.fields.api_key.value, 'in-memory-only');
  assert.equal(app.state.files[0], file);
});

test('removing material requires the material step again without discarding model settings', () => {
  const app = createApp();
  app.state.sourceMode = 'manual';
  app.context.updateSourceCount();
  assert.equal(app.state.setupStage, 'sources');
  assert.equal(app.node('#run-button').hidden, true);
  assert.equal(app.fields.api_model.value, 'test-model');
});

test('opening an existing run leaves model settings available for continuation without enabling a new run', () => {
  const app = createApp();
  app.state.currentRunId = 'existing-run';
  app.context.initializeSetupFlow({ focus: false });
  assert.equal(app.node('#model-configuration').hidden, false);
  assert.equal(app.node('#run-button').hidden, true);
  assert.equal(app.focused(), undefined);
});
