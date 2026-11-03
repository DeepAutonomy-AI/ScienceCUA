const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function renderer(runs = []) {
  const fields = new Map();
  const field = selector => {
    if (!fields.has(selector)) fields.set(selector, {
      attributes: {}, dataset: {}, style: { setProperty() {} }, classList: { toggle() {} },
      setAttribute(name, value) { this.attributes[name] = value; },
      querySelector() { return field(`${selector} span`); },
      querySelectorAll() { return []; },
    });
    return fields.get(selector);
  };
  const context = vm.createContext({
    document: { addEventListener() {}, querySelector: field, querySelectorAll: () => [] },
    fetch: async () => ({ json: async () => ({ runs }) }),
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  const app = vm.runInContext('appState', context);
  return { context, app, field };
}

const artifacts = () => ({
  outcome: 'degraded',
  skills: [{ skill_id: 'SK1', title: 'Display a vector map', description: 'Render vector features.', trajectory_ids: ['TR1'] }],
  references: [{ reference_id: 'R1', title: 'Vector display parameters' }],
});

test('one validated Skill and Reference make a finished run successful despite source and trajectory gaps', () => {
  const { context } = renderer();
  const state = { status: 'degraded', phase: 'degraded',
    source_results: [{ status: 'degraded' }], task_trajectory_stage_result: { status: 'degraded' } };
  const result = artifacts();
  const before = JSON.stringify({ state, result });
  assert.equal(context.getRunDisplayStatus(state, result), 'succeeded');
  assert.equal(context.getRunStageLabel(state, 'succeeded'), 'Run completed');
  assert.equal(JSON.stringify({ state, result }), before);
});

test('both deliverables are required, and explicit empty results override old state counts', () => {
  const { context } = renderer();
  const state = { status: 'completed', skill_count: 22, reference_count: 24 };
  for (const missing of ['skills', 'references']) {
    const result = { ...artifacts(), outcome: 'succeeded', [missing]: [] };
    assert.equal(context.getRunDisplayStatus(state, result), 'degraded', missing);
  }
  assert.equal(context.getRunDisplayStatus({ status: 'degraded' }), 'degraded');
});

test('proposals and legacy placeholder Skills do not count as valid deliverables', () => {
  const { context } = renderer();
  assert.equal(context.getRunDisplayStatus({ status: 'degraded',
    skill_candidates: artifacts().skills, reference_candidates: artifacts().references }), 'degraded');
  const result = { ...artifacts(), skills: [{ title: 'Chapter 1 Introduction' }] };
  assert.equal(context.getRunDisplayStatus({ status: 'degraded', skill_count: 1 }, result), 'degraded');
});

test('partial output never overrides running, paused, failed or stopped states', () => {
  const { context } = renderer();
  for (const status of ['queued', 'running', 'cancelling', 'waiting_for_input', 'paused', 'failed', 'cancelled']) {
    assert.equal(context.getRunDisplayStatus({ status, skill_count: 1, reference_count: 1 }, artifacts()), status);
  }
});

test('recent runs apply the same criterion using validated state artifacts and counts', async () => {
  const runs = [
    { run_id: 'run-complete', phase: 'degraded', status: 'degraded', skill_count: 1, reference_count: 1,
      validated_skills: artifacts().skills },
    { run_id: 'run-no-reference', phase: 'degraded', status: 'degraded', skill_count: 1, reference_count: 0 },
    { run_id: 'run-placeholder', phase: 'degraded', status: 'degraded', skill_count: 1, reference_count: 1,
      validated_skills: [{ title: 'Chapter 1 Introduction' }] },
  ];
  const { context, field } = renderer(runs);
  await context.refreshRuns();
  const buttons = field('#recent-list').innerHTML.match(/<button[\s\S]*?<\/button>/g);
  assert.match(buttons[0], /Run completed<\/small>/);
  assert.match(buttons[0], /status-badge completed">Completed/);
  for (const button of buttons.slice(1)) assert.match(button, /status-badge degraded">With gaps/);
  assert.equal(runs[0].status, 'degraded');
});

test('the result title and progress status agree in Chinese without changing backend state', () => {
  const { context, app, field } = renderer();
  for (const name of ['updateTabIndicator', 'renderStageRail', 'renderSources', 'renderEvidence', 'renderReferences',
    'renderTasksAndTrajectories', 'renderSkills', 'renderAudit', 'renderSourceSelectionState', 'refreshIcons']) {
    context[name] = () => {};
  }
  app.language = 'Chinese';
  app.runPayload = { state: { status: 'degraded', phase: 'degraded' }, result: artifacts() };
  context.renderRun();
  assert.equal(field('#run-status').textContent, '已完成');
  assert.equal(field('#run-status').className, 'status-badge completed');
  assert.equal(field('#stage-label').textContent, '任务已完成');
  assert.equal(field('#progress-status').textContent, '已完成');
  assert.equal(field('.progress-band').dataset.status, 'succeeded');
  assert.equal(field('.progress-track').attributes['aria-valuetext'], '100% · 任务已完成');
  assert.equal(app.runPayload.state.status, 'degraded');
  assert.equal(app.runPayload.result.outcome, 'degraded');
});

test('the Sources table retains its four metadata columns and omits status, coverage and excerpt cells', () => {
  const { context, app, field } = renderer();
  app.runPayload = { state: {}, result: { snapshot: { source_results: [{
    source_id: 'SRC1', display_name: 'GRASS GIS manual', modality: 'html', status: 'degraded',
    manifest: { resolved_version: '8.4', source_origin: 'user_upload' },
    coverage: { discovered_parts: 875 }, evidence: [{ evidence_id: 'E1' }],
  }] } } };
  context.renderSources();
  const cells = field('#source-table-body').innerHTML.match(/<td(?:\s[^>]*)?>[\s\S]*?<\/td>/g);
  assert.equal(cells.length, 4);
  assert.match(cells[0], /GRASS GIS manual/);
  assert.match(cells[1], /html/);
  assert.match(cells[2], /8\.4/);
  assert.doesNotMatch(field('#source-table-body').innerHTML, /With gaps|coverage-cell|875/);
  const html = fs.readFileSync(path.resolve(__dirname, '../../app/static/index.html'), 'utf8');
  const headers = html.match(/<table class="source-table">[\s\S]*?<thead>([\s\S]*?)<\/thead>/)[1];
  assert.deepEqual([...headers.matchAll(/data-i18n="([^"]+)"/g)].map(match => match[1]),
    ['Source', 'Modality', 'Version', 'Origin']);
});
