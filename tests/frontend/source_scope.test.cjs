const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
const source = fs.readFileSync(appPath, 'utf8');

function createApp() {
  const nodes = new Map();
  let sectionInputs = [];
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', innerHTML: '', hidden: false, disabled: false, open: false, checked: false,
      handlers: new Map(), attributes: new Map(), classList: { add() {}, remove() {}, toggle() {} },
      querySelector: (child) => node(`${selector} ${child}`), querySelectorAll: () => [],
      setAttribute(name, value) { this.attributes.set(name, value); }, removeAttribute(name) { this.attributes.delete(name); },
      addEventListener(type, handler) { this.handlers.set(type, handler); },
      showModal() { this.open = true; }, close() { this.open = false; }, scrollIntoView() {},
    });
    return nodes.get(selector);
  };
  const query = (selector) => {
    if (selector === "#source-scope-form input[name='scope_mode']:checked") {
      return node('#scope-mode-selected').checked ? node('#scope-mode-selected') : node('#scope-mode-full');
    }
    return node(selector);
  };
  const requests = [];
  const context = vm.createContext({
    document: { addEventListener() {}, querySelector: query, querySelectorAll: (selector) => {
      if (selector.startsWith('#source-scope-sections input')) {
        return selector.endsWith(':checked') ? sectionInputs.filter(input => input.checked) : sectionInputs;
      }
      return [];
    } }, window: {}, clearTimeout() {}, setTimeout() {},
    fetch: async (url, options) => {
      requests.push({ url, body: JSON.parse(options.body) });
      return { ok: true, json: async () => ({ status: 'running' }) };
    },
  });
  vm.runInContext(source, context, { filename: appPath });
  vm.runInContext('pollRun = async () => {};', context);
  const state = vm.runInContext('appState', context);
  state.currentRunId = 'scope-run';
  state.runPayload = {
    state: { phase: 'awaiting_scope_selection', status: 'waiting_for_input', active_task: false },
    source_scope: { outline_artifact_id: 'outline-1', source_id: 'SRC1', modality: 'pdf', title: 'Manual',
      outline_available: true, warnings: [], sections: [
        { section_id: 's1', title: '<Navigation>', page_start: 3, page_end: 9 },
        { section_id: 's2', title: 'Display', page_start: 10, page_end: 22 },
      ] },
  };
  sectionInputs = state.runPayload.source_scope.sections.map(section => ({ value: section.section_id, checked: false, disabled: false }));
  node('#scope-mode-full').value = 'full';
  node('#scope-mode-selected').value = 'selected';
  const submit = () => context.submitScopeSelection({ preventDefault() {} });
  return { context, state, node, requests, submit, sectionInputs };
}

test('scope selection defaults to the whole material and escapes original chapter titles', () => {
  const { context, state, node, sectionInputs } = createApp();
  context.showScopeChoices();
  assert.equal(node('#source-scope-dialog').open, true);
  assert.equal(context.currentScopeDraft().mode, 'full');
  assert.equal(node('#scope-mode-full').checked, true);
  assert.match(node('#source-scope-sections').innerHTML, /&lt;Navigation&gt;/);
  assert.doesNotMatch(node('#source-scope-sections').innerHTML, /<Navigation>/);
  assert.match(node('#source-scope-sections').innerHTML, /Pages 3–9/);
  assert.equal(node('#source-scope-budget').hidden, true);
  assert.equal(sectionInputs.every(input => input.disabled), true);
  assert.equal(state.scopeOutlineArtifactId, 'outline-1');
});

test('reviewed trees show first-level choices only with measured coverage and post-review guidance', () => {
  const { context, state, node } = createApp();
  const scope = state.runPayload.source_scope;
  Object.assign(scope, { schema_version: 'source-outline.v2', structure_review_complete: true,
    modality: 'html', acquisition_coverage: { page_count: 13, acquisition_complete: false },
    warnings: ['One linked page could not be fetched.'] });
  Object.assign(scope.sections[0], { depth: 0, direct_char_count: 10, descendant_char_count: 20 });
  Object.assign(scope.sections[1], { depth: 1, direct_char_count: 20, descendant_char_count: 0 });
  context.renderScopeSelectionState(state.runPayload.state);
  assert.match(node('#source-scope-sections').innerHTML, /30 parsed characters/);
  assert.doesNotMatch(node('#source-scope-sections').innerHTML, /value="s2"/);
  assert.equal((node('#source-scope-sections').innerHTML.match(/type="checkbox"/g) || []).length, 1);
  assert.match(node('#source-scope-budget').textContent, /13 pages of material parsed/);
  assert.match(node('#scope-selection-notice-text').textContent, /Parsing and structure review are complete/);
  assert.match(node('#source-scope-warnings').textContent, /could not be fully collected or processed/);
  assert.doesNotMatch(node('#source-scope-budget').textContent, /will be processed/);
});

test('nested chapter IDs cannot be submitted even from an old outline or draft', async () => {
  const { context, state, requests, node, submit } = createApp();
  state.runPayload.source_scope.sections.push({ section_id: 's1-child', title: 'Nested', depth: 1 },
    { section_id: 's1-grandchild', title: 'Deeply nested', depth: 2 });
  context.showScopeChoices();
  assert.equal((node('#source-scope-sections').innerHTML.match(/type="checkbox"/g) || []).length, 2);
  assert.doesNotMatch(node('#source-scope-sections').innerHTML, /Nested|Deeply nested/);
  const draft = context.currentScopeDraft();
  draft.mode = 'selected';
  draft.section_ids = ['s1-child'];
  await submit();
  assert.equal(requests.length, 0);
  assert.match(node('#source-scope-error').textContent, /Choose at least one chapter/);
  draft.section_ids = ['s1'];
  await submit();
  assert.deepEqual(requests[0].body.section_ids, ['s1']);
});

test('uncommitted nested selections are cleared when restoring an old draft', () => {
  const { context, state } = createApp();
  const scope = state.runPayload.source_scope;
  scope.sections.push({ section_id: 'child', title: 'Nested', depth: 1 });
  scope.selection = { mode: 'selected', section_ids: ['child', 's1'] };
  assert.deepEqual(Array.from(context.currentScopeDraft().section_ids), ['s1']);
});

test('historical committed nested choices resume without exposing new nested options', async () => {
  const { context, state, requests, node, submit } = createApp();
  state.runPayload.state = { phase: 'scope_selected', status: 'running', active_task: false };
  state.runPayload.source_scope.sections.push({ section_id: 'child', title: 'Nested', depth: 1 });
  state.runPayload.source_scope.selection = { mode: 'selected', section_ids: ['child'] };
  context.showScopeChoices();
  assert.doesNotMatch(node('#source-scope-sections').innerHTML, /value="child"/);
  assert.equal(node('#scope-mode-selected').disabled, true);
  await submit();
  assert.deepEqual(requests[0].body.section_ids, ['child']);
});

test('scope mode and chapter checkboxes preserve multiple choices across dismiss and reopen', () => {
  const { context, node, sectionInputs } = createApp();
  context.bindSourceDialogs();
  context.showScopeChoices();
  node('#scope-mode-selected').checked = true;
  sectionInputs.forEach(input => { input.checked = true; });
  node('#source-scope-form').handlers.get('change')();
  assert.equal(context.currentScopeDraft().mode, 'selected');
  assert.deepEqual(Array.from(context.currentScopeDraft().section_ids), ['s1', 's2']);
  assert.equal(sectionInputs.every(input => !input.disabled), true);
  node('#source-scope-close').handlers.get('click')();
  node('#open-source-scope').handlers.get('click')();
  assert.equal(node('#source-scope-dialog').open, true);
  assert.equal((node('#source-scope-sections').innerHTML.match(/ checked/g) || []).length, 2);
  assert.match(node('#source-scope-selection-count').textContent, /2 chapters/);
});

test('chapter outline opens only once per run and can be reopened deliberately', () => {
  const { context, state, node } = createApp();
  context.renderSourceSelectionState(state.runPayload.state);
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-selection-notice').hidden, true);
  assert.equal(node('#source-candidates-dialog').open, false);
  assert.equal(node('#scope-selection-notice').hidden, false);
  assert.equal(node('#source-scope-dialog').open, true);
  node('#source-scope-dialog').close();
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-scope-dialog').open, false);
  context.showScopeChoices();
  assert.equal(node('#source-scope-dialog').open, true);
});

test('an open source-mode dialog defers the automatic chapter prompt until the next poll', () => {
  const { context, state, node } = createApp();
  node('#source-mode-dialog').open = true;
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-scope-dialog').open, false);
  node('#source-mode-dialog').open = false;
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-scope-dialog').open, true);
});

test('page estimates remain incomplete and the linked-page limit excludes the starting page', () => {
  const { context, state, node } = createApp();
  Object.assign(state.runPayload.source_scope, { modality: 'html', max_child_pages: 80, page_budget_scope: 'child_pages_only',
    sections: [{ section_id: 's1', title: 'Commands', known_page_count: 12 }, { section_id: 's2', title: 'Examples' }] });
  context.showScopeChoices();
  assert.equal(node('#source-scope-budget').hidden, false);
  assert.match(node('#source-scope-budget').textContent, /up to 80 linked pages/);
  assert.match(node('#source-scope-budget').textContent, /in addition to the starting page/);
  const html = node('#source-scope-sections').innerHTML;
  assert.match(html, /12 known linked pages; more may be found/);
  assert.match(html, /Page links will be expanded after selection/);
  assert.doesNotMatch(html, /undefined|0 known linked pages/);
});

test('unreliable outlines offer full material with a clear explanation', () => {
  const { context, state, node } = createApp();
  Object.assign(state.runPayload.source_scope, { outline_available: false, sections: [], warnings: ['Source warning'] });
  context.showScopeChoices();
  assert.equal(node('#scope-mode-selected').disabled, true);
  assert.equal(node('#scope-mode-full').checked, true);
  assert.equal(node('#source-scope-unavailable').hidden, false);
  assert.match(node('#source-scope-warnings').textContent, /Some material could not be fully collected or processed/);
});

test('hundreds of acquisition diagnostics become one localized notice without changing saved details', () => {
  const { context, state, node } = createApp();
  const scope = state.runPayload.source_scope;
  Object.assign(scope, { modality: 'html', structure_review_complete: true,
    acquisition_coverage: { page_count: 81, acquisition_complete: false, truncated: true },
    warnings: Array.from({ length: 500 }, (_, i) =>
      `Guided HTML acquisition budget_exhausted: https://example.test/manual/${i}.html (child_page_budget)`) });
  const original = JSON.stringify(scope);
  for (const language of ['English', 'Chinese']) {
    state.language = language;
    context.showScopeChoices();
    const text = node('#source-scope-warnings').textContent;
    assert.equal(node('#source-scope-warnings').hidden, false);
    assert.ok(text.length < 250);
    assert.doesNotMatch(text, /https?:|\.html|budget_exhausted|child_page_budget|\n/);
    assert.match(text, language === 'Chinese' ? /处理上限.*继续生成/ : /collection limit.*continue/);
    assert.doesNotMatch(node('#source-scope-budget').textContent, /HTML|gaps are listed/);
    assert.equal(JSON.stringify(scope), original);
  }
});

test('structured gaps and historical diagnostics both retain an honest compact coverage warning', () => {
  const { context, state, node } = createApp();
  const scope = state.runPayload.source_scope;
  for (const status of ['budget_exhausted', 'depth_exhausted']) {
    Object.assign(scope, { warnings: [], acquisition_coverage: { navigation_gaps: [{ status }] } });
    context.showScopeChoices();
    assert.match(node('#source-scope-warnings').textContent, /collection limit/);
    Object.assign(scope, { acquisition_coverage: {},
      warnings: [`Guided HTML acquisition ${status}: https://example.test/path.html (internal_reason)`] });
    context.showScopeChoices();
    assert.match(node('#source-scope-warnings').textContent, /collection limit/);
    assert.doesNotMatch(node('#source-scope-warnings').textContent, /example\.test|internal_reason/);
  }
  Object.assign(scope, { warnings: [], acquisition_coverage: { acquisition_complete: false } });
  context.showScopeChoices();
  assert.equal(node('#source-scope-warnings').hidden, false);
  assert.match(node('#source-scope-warnings').textContent, /could not be fully collected/);
  Object.assign(scope, { warnings: ['Private path C:\\work\\source.html: parser warning'], acquisition_coverage: {} });
  context.showScopeChoices();
  assert.doesNotMatch(node('#source-scope-warnings').textContent, /Private|source\.html|parser warning/);
});

test('complete material clears stale warnings from a previously incomplete outline', () => {
  const { context, state, node } = createApp();
  const scope = state.runPayload.source_scope;
  scope.warnings = ['A linked page was unavailable'];
  context.showScopeChoices();
  assert.equal(node('#source-scope-warnings').hidden, false);
  scope.warnings = [];
  scope.acquisition_coverage = { acquisition_complete: true };
  context.renderScopeChoices();
  assert.equal(node('#source-scope-warnings').hidden, true);
  assert.equal(node('#source-scope-warnings').textContent, '');
});

test('selected scope requires at least one chapter and rejects unknown chapter IDs', async () => {
  const { context, node, requests, submit } = createApp();
  context.showScopeChoices();
  const draft = context.currentScopeDraft();
  draft.mode = 'selected';
  await submit();
  assert.equal(requests.length, 0);
  assert.match(node('#source-scope-error').textContent, /at least one chapter/);
  draft.section_ids = ['stale-section'];
  await submit();
  assert.equal(requests.length, 0);
});

test('full scope clears draft chapter IDs from the API request', async () => {
  const { context, node, requests, submit } = createApp();
  context.showScopeChoices();
  context.currentScopeDraft().section_ids = ['s1'];
  await submit();
  assert.deepEqual(requests, [{ url: '/api/runs/scope-run/select-scope',
    body: { outline_artifact_id: 'outline-1', mode: 'full', section_ids: [], api_key: '' } }]);
  assert.equal(node('#source-scope-dialog').open, false);
});

test('selected chapters submit once with the outline ID and current API key', async () => {
  const { context, state, node, submit } = createApp();
  context.showScopeChoices();
  Object.assign(context.currentScopeDraft(), { mode: 'selected', section_ids: ['s1', 's2', 's1'] });
  node("#run-form input[name='api_key']").value = 'memory-only-key';
  const requests = [];
  let release;
  context.fetch = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    await new Promise(resolve => { release = resolve; });
    return { ok: true, json: async () => ({ status: 'running' }) };
  };
  const first = submit();
  await submit();
  assert.equal(requests.length, 1);
  assert.equal(node('#source-scope-continue').disabled, true);
  assert.deepEqual(requests[0], { url: '/api/runs/scope-run/select-scope',
    body: { outline_artifact_id: 'outline-1', mode: 'selected', section_ids: ['s1', 's2'], api_key: 'memory-only-key' } });
  release();
  await first;
  assert.equal(state.scopeRequestPending, false);
});

test('continuation waits for the outline task to finish and rejects stale dialogs', async () => {
  const { context, state, node, requests, submit } = createApp();
  state.runPayload.state.active_task = true;
  context.showScopeChoices();
  assert.equal(node('#source-scope-continue').disabled, true);
  await submit();
  assert.equal(requests.length, 0);
  state.runPayload.state.active_task = false;
  state.runPayload.source_scope.outline_artifact_id = 'outline-2';
  await submit();
  assert.equal(requests.length, 0);
});

test('restored scope selection requests a missing API key and lets users supply it in the dialog', async () => {
  const { context, state, node, requests, submit } = createApp();
  state.language = 'Chinese';
  state.runPayload.state.requires_api_key = true;
  context.bindSourceDialogs();
  context.showScopeChoices();
  assert.equal(node('#scope-api-key-field').hidden, false);
  assert.equal(node('#scope-api-key').required, true);
  await submit();
  assert.equal(requests.length, 0);
  assert.match(node('#source-scope-error').textContent, /API Key/);
  assert.equal(node('#model-configuration').open, true);
  node('#scope-api-key').handlers.get('input')({ target: { value: 'restored-key' } });
  await submit();
  assert.equal(requests[0].body.api_key, 'restored-key');
});

test('scope chapters remain untranslated while labels and estimates use the page language', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  context.showScopeChoices();
  assert.match(node('#source-scope-sections').innerHTML, /Navigation|Display/);
  assert.match(node('#source-scope-sections').innerHTML, /第 3–9 页/);
  assert.equal(context.stageName('awaiting_scope_selection'), '等待选择提取范围');
  assert.equal(context.t('Confirm scope and continue'), '确认范围并继续');
});

test('scope preview and waiting stay within the Source stage without showing result skeletons', () => {
  const { context, state } = createApp();
  for (const source_mode of ['manual', 'discovery']) {
    for (const phase of ['previewing_scope', 'awaiting_scope_selection', 'scope_selected']) {
      const progress = context.getPipelineProgress({ phase, source_mode, status: phase === 'awaiting_scope_selection' ? 'waiting_for_input' : 'running' });
      const index = source_mode === 'discovery' ? 1 : 0;
      assert.equal(progress.stages[index].active, true);
      assert.equal(progress.percent, index ? 25 : 10);
      if (phase === 'awaiting_scope_selection') assert.equal(progress.stages[index].paused, true);
    }
  }
  for (const phase of ['previewing_scope', 'awaiting_scope_selection']) {
    state.runPayload.state = { phase, status: 'running' };
    for (const section of ['evidence', 'reference', 'tasks', 'skills']) assert.equal(context.shouldShowResultSkeleton(section), false);
  }
});

test('scope prompt closes after continuation or cancellation', () => {
  const { context, state, node } = createApp();
  context.showScopeChoices();
  state.runPayload.state.status = 'cancelled';
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-scope-dialog').open, false);
  assert.equal(node('#scope-selection-notice').hidden, true);
  assert.equal(context.isAwaitingScope(), false);
});

for (const mode of ['full', 'selected']) {
  test(`a committed ${mode} scope can resume after a restart without changing its selection`, async () => {
    const { context, state, node, requests, sectionInputs, submit } = createApp();
    const section_ids = mode === 'selected' ? ['s2'] : [];
    state.runPayload.state = { phase: 'scope_selected', status: 'running', active_task: false, requires_api_key: true };
    state.runPayload.source_scope.selection = { mode, section_ids };
    context.bindSourceDialogs();
    context.renderScopeSelectionState(state.runPayload.state);
    assert.equal(node('#source-scope-dialog').open, true);
    assert.equal(node('#scope-selection-notice').hidden, false);
    assert.match(node('#scope-selection-notice-text').textContent, /scope is saved/);
    assert.equal(node('#open-source-scope').textContent, 'Continue extraction');
    assert.equal(node('#scope-mode-full').disabled, true);
    assert.equal(node('#scope-mode-selected').disabled, true);
    assert.equal(sectionInputs.every(input => input.disabled), true);
    assert.equal(node('#source-scope-continue').textContent, 'Continue extraction');
    // Neither an obsolete local draft nor a programmatic change can broaden the saved choice.
    Object.assign(context.currentScopeDraft(), { mode: 'selected', section_ids: ['s1'] });
    node('#scope-mode-selected').checked = true;
    sectionInputs.forEach(input => { input.checked = true; });
    node('#source-scope-form').handlers.get('change')();
    node('#scope-api-key').handlers.get('input')({ target: { value: 'restart-key' } });
    await submit();
    assert.deepEqual(requests[0].body, { outline_artifact_id: 'outline-1', mode, section_ids, api_key: 'restart-key' });
    const progress = context.getPipelineProgress(state.runPayload.state);
    assert.equal(progress.stages[0].paused, true);
    assert.equal(progress.stages[0].label, 'Waiting to continue');
  });
}

test('a running scope continuation does not offer another resume action', () => {
  const { context, state, node } = createApp();
  state.runPayload.state = { phase: 'scope_selected', status: 'running', active_task: true };
  state.runPayload.source_scope.selection = { mode: 'selected', section_ids: ['s1'] };
  context.renderScopeSelectionState(state.runPayload.state);
  assert.equal(node('#source-scope-dialog').open, false);
  assert.equal(node('#scope-selection-notice').hidden, true);
});

test('disabled chapter selection clears an old draft and submits only the complete manual', async () => {
  const { context, state, node, requests, submit } = createApp();
  Object.assign(context.currentScopeDraft(), { mode: 'selected', section_ids: ['s1'] });
  state.runPayload.source_scope.selection_enabled = false;
  context.showScopeChoices();
  assert.equal(context.currentScopeDraft().mode, 'full');
  assert.equal(node('#scope-mode-selected').disabled, true);
  assert.equal(node('#scope-mode-full').checked, true);
  await submit();
  assert.equal(requests[0].body.mode, 'full');
  assert.deepEqual(requests[0].body.section_ids, []);
});

test('turning selection off does not broaden a committed chapter scope on resume', async () => {
  const { context, state, node, requests, submit } = createApp();
  state.runPayload.source_scope.selection_enabled = false;
  state.runPayload.state = { phase: 'scope_selected', status: 'running', active_task: false };
  state.runPayload.source_scope.selection = { mode: 'selected', section_ids: ['s2'] };
  context.showScopeChoices();
  assert.equal(context.currentScopeDraft().mode, 'selected');
  assert.equal(node('#scope-mode-selected').disabled, true);
  await submit();
  assert.equal(requests[0].body.mode, 'selected');
  assert.deepEqual(requests[0].body.section_ids, ['s2']);
});
