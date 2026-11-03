const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function createApp() {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', innerHTML: '', hidden: false, disabled: false,
      querySelectorAll: () => [], querySelector: child => node(`${selector} ${child}`),
      setAttribute(name, value) { this[name] = value; },
    });
    return nodes.get(selector);
  };
  class FormData {
    constructor(form) { this.values = new Map(Object.entries(form?.values || {})); }
    get(key) { return this.values.get(key); }
    set(key, value) { this.values.set(key, value); }
    delete(key) { this.values.delete(key); }
    append(key, value) { this.values.set(key, value); }
  }
  const requests = [];
  let sequence = 0;
  const context = vm.createContext({ window: {}, FormData, URL,
    document: { addEventListener() {}, querySelector: node, querySelectorAll: () => [] },
    crypto: { randomUUID: () => `draft-${++sequence}` },
    fetch: async (url, options) => { requests.push({ url, form: options.body }); return { ok: true, json: async () => ({ run_id: 'mock' }) }; },
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  vm.runInContext('selectRun = refreshRuns = async () => {};', context);
  const state = vm.runInContext('appState', context);
  state.setupStage = 'model';
  state.sourceModeChosen = true;
  node("#run-form input[name='target_version']").value = '1.9';
  const submit = () => {
    // These tests cover the final payload after the user completes the setup steps.
    state.setupStage = 'model';
    return context.submitRun({ preventDefault() {}, target: { values: { software_name: 'ChimeraX' } } });
  };
  return { context, state, node, requests, submit };
}

test('switching between upload and link preserves both drafts and their versions', () => {
  const { context, state } = createApp();
  const file = { key: 'file-a', file: { name: 'manual.pdf' }, version: '1.8' };
  state.files = [file];
  context.switchManualSourceKind('url');
  assert.equal(state.files.length, 0);
  assert.equal(state.urls.length, 1);
  state.urls[0].url = 'https://example.org/manual';
  state.urls[0].version = '1.9';
  context.switchManualSourceKind('file');
  assert.equal(state.files[0], file);
  assert.equal(state.urls.length, 0);
  context.switchManualSourceKind('url');
  assert.equal(state.urls[0].url, 'https://example.org/manual');
  assert.equal(state.urls[0].version, '1.9');
  assert.equal(state.files.length, 0);
});

test('upload submission excludes the saved link draft', async () => {
  const { context, state, submit, requests } = createApp();
  context.switchManualSourceKind('url');
  state.urls[0].url = 'https://example.org/old-manual';
  context.switchManualSourceKind('file');
  const file = { name: 'manual.pdf' };
  state.files = [{ key: 'file-a', file, version: '1.9' }];
  await submit();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].form.get('files'), file);
  assert.equal(requests[0].form.get('urls_json'), '[]');
  assert.equal(requests[0].form.get('file_versions_json'), '["1.9"]');
});

test('link submission excludes the saved file draft and discovery excludes both', async () => {
  const { context, state, submit, requests } = createApp();
  state.files = [{ key: 'file-a', file: { name: 'manual.pdf' }, version: '1.8' }];
  context.switchManualSourceKind('url');
  state.urls[0].url = 'file:///C:/manual.html';
  state.urls[0].version = '1.9';
  await submit();
  assert.equal(requests[0].form.get('files'), undefined);
  assert.equal(requests[0].form.get('file_versions_json'), '[]');
  assert.deepEqual(JSON.parse(requests[0].form.get('urls_json')), [{ url: 'file:///C:/manual.html', user_declared_version: '1.9' }]);
  state.sourceMode = 'discovery';
  await submit();
  assert.equal(requests[1].form.get('files'), undefined);
  assert.equal(requests[1].form.get('urls_json'), '[]');
});

test('cancelled, unsupported and multiple-file replacement preserves the selected file', () => {
  const { context, state } = createApp();
  const original = { key: 'file-a', file: { name: 'manual.pdf' }, version: '1.8' };
  state.files = [original];
  for (const files of [[], [{ name: 'notes.txt' }], [{ name: 'one.pdf' }, { name: 'two.pdf' }]]) {
    context.addFiles(files);
    assert.equal(state.files[0], original);
  }
  context.addFiles([{ name: 'replacement.html', size: 1024, lastModified: 1 }]);
  assert.equal(state.files[0].file.name, 'replacement.html');
});

test('unknown versions are blank in the UI and model summary excludes API credentials', () => {
  const { context, node } = createApp();
  for (const version of ['unknown', ' Unknown ', '', null]) assert.equal(context.sourceVersionInputValue(version), '');
  assert.equal(context.sourceVersionInputValue('1.9'), '1.9');
  node("#api-fields [name='api_model']").value = 'my-model';
  node("#api-fields [name='api_base_url']").value = 'https://example.org/v1';
  node("#api-fields [name='api_key']").value = 'do-not-display';
  context.renderModelConfigurationSummary();
  assert.equal(node('#model-summary').textContent, 'my-model');
  assert.equal(node('#model-configuration-state').textContent, 'Configured');
  assert.doesNotMatch(node('#model-summary').textContent, /do-not-display/);
});
