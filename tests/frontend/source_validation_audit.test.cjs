const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
const source = fs.readFileSync(appPath, 'utf8');
const reportId = 'art_0123456789abcdef01234567';

function createApp() {
  const nodes = new Map();
  function node(selector) {
    if (!nodes.has(selector)) nodes.set(selector, {
      innerHTML: '', checked: false, scrollTop: 0, scrollHeight: 100,
    });
    return nodes.get(selector);
  }
  const context = vm.createContext({
    document: { addEventListener() {}, querySelector: node, querySelectorAll: () => [] },
    window: {},
  });
  vm.runInContext(source, context, { filename: appPath });
  const state = vm.runInContext('appState', context);
  state.currentRunId = 'run_current';
  return { context, state, node };
}

function validationEvent(overrides = {}) {
  return {
    event_type: 'source_validation_completed', source_id: 'SRC1', run_id: 'run_original',
    timestamp: '2026-10-14T04:51:12Z', level: 'warning',
    payload: {
      status: 'degraded', input_evidence_count: 291, evidence_count: 144,
      rejected_evidence_count: 147, error_count: 147, validation_artifact_id: reportId,
    },
    ...overrides,
  };
}

test('source validation summary exposes accepted, input, and rejected Evidence counts', () => {
  const { context } = createApp();
  assert.equal(context.auditSummary(validationEvent()), 'SRC1 · Evidence: 144/291 accepted; 147 rejected');
  assert.equal(context.auditSummary(validationEvent({ agent: 'html_source_review' })),
    'html_source_review · Evidence: 144/291 accepted; 147 rejected');
});

test('source validation counts and report caption follow Chinese UI language', () => {
  const { context, state } = createApp();
  state.language = 'Chinese';
  const summary = context.auditSummary(validationEvent());
  assert.match(summary, /144\/291/);
  assert.match(summary, /147/);
  assert.match(summary, /[\u3400-\u9fff]/);
  assert.doesNotMatch(summary, /accepted|rejected/);
  const link = context.renderValidationReportLink(validationEvent());
  assert.match(link, /[\u3400-\u9fff]/);
  assert.doesNotMatch(link, /View validation report/);
});

test('validation report link uses event run scope and encodes run URL segment', () => {
  const { context } = createApp();
  const runId = 'run/original?x=1&y=2';
  const link = context.renderValidationReportLink(validationEvent({ run_id: runId }));
  assert.match(link, new RegExp(`href="/api/runs/${encodeURIComponent(runId)}/artifacts/${reportId}"`));
  assert.match(link, /target="_blank"/);
  assert.match(link, /rel="noopener"/);
  assert.match(link, />View validation report<\/a>/);
  assert.doesNotMatch(link, /run_current/);
});

test('validation report link falls back to current run only when event run is absent', () => {
  const { context, state } = createApp();
  const event = validationEvent({ run_id: undefined });
  assert.match(context.renderValidationReportLink(event), /\/api\/runs\/run_current\/artifacts\//);
  state.currentRunId = null;
  assert.equal(context.renderValidationReportLink(event), '');
});

test('missing and malformed report Artifact IDs do not create a link', () => {
  const { context } = createApp();
  for (const id of [undefined, null, '', 'art_bad', `art_${'g'.repeat(24)}`,
    `${reportId}/../../private`, `${reportId}" onclick="alert(1)`, reportId.toUpperCase()]) {
    assert.equal(context.renderValidationReportLink(validationEvent({
      payload: { validation_artifact_id: id },
    })), '', `Unexpected link for ${String(id)}`);
  }
  assert.equal(context.renderValidationReportLink(validationEvent({ payload: undefined })), '');
  assert.equal(context.renderValidationReportLink(validationEvent({ event_type: 'source_result_accepted' })), '');
});

test('Errors audit filter displays validation warnings and errors with counts and report links', () => {
  const { context, state, node } = createApp();
  state.auditFilter = 'error';
  state.events = [
    validationEvent({ source_id: 'SRCinfo', level: 'info' }),
    validationEvent({ source_id: 'SRCwarning', level: 'warning' }),
    validationEvent({ source_id: 'SRCerror', level: 'error' }),
  ];
  context.renderAudit();
  const html = node('#audit-list').innerHTML;
  assert.doesNotMatch(html, /SRCinfo/);
  assert.match(html, /SRCwarning · Evidence: 144\/291 accepted; 147 rejected/);
  assert.match(html, /SRCerror · Evidence: 144\/291 accepted; 147 rejected/);
  assert.equal((html.match(/View validation report/g) || []).length, 2);
  assert.match(html, /<\/pre>\s*<a[^>]+>View validation report<\/a>\s*<\/details>/);
  assert.match(html, /class="audit-event\s+error"/);
});

test('validation event source identity is escaped when rendering an audit summary', () => {
  const { context, state, node } = createApp();
  state.events = [validationEvent({ source_id: '<img src=x onerror=alert(1)>' })];
  context.renderAudit();
  const html = node('#audit-list').innerHTML;
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt; · Evidence/);
  assert.doesNotMatch(html, /<img src=x/);
});

test('Model audit filter shows waiting duration and retry reason in the visible summary', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.auditFilter = 'model';
  state.events = [
    {event_type: 'knowledge_tool_call', timestamp: '2026-10-14T14:50:00Z', payload: {tool: 'read_evidence'}},
    {event_type: 'model_waiting', agent: 'reference_build', timestamp: '2026-10-14T14:50:00Z', level: 'info',
      payload: {status: 'waiting_for_model', elapsed_seconds: 121.4, seconds_since_event: 100.2}},
    {event_type: 'model_retry_scheduled', agent: 'reference_build', timestamp: '2026-10-14T14:50:00Z', level: 'warning',
      payload: {status: 'retry_wait', retry_number: 2, max_retries: 5, delay_seconds: 3, message: 'HTTP 503 <temporary>'}},
  ];
  context.renderAudit();
  const html = node('#audit-list').innerHTML;
  assert.match(html, /<summary>reference_build · 模型会话等待：已用时 121 秒，100 秒无新活动<\/summary>/);
  assert.match(html, /<summary>reference_build · 模型重试 2\/5，3 秒后重试：HTTP 503 &lt;temporary&gt;<\/summary>/);
  assert.doesNotMatch(html, /knowledge_tool_call|<temporary>/);
});

test('Errors audit filter shows retry warnings and a non-retryable model error', () => {
  const { context, state, node } = createApp();
  state.auditFilter = 'error';
  state.events = [
    {event_type: 'model_waiting', timestamp: '2026-10-14T14:50:00Z', level: 'info', payload: {}},
    {event_type: 'model_retry_scheduled', timestamp: '2026-10-14T14:50:00Z', level: 'warning',
      payload: {retry_number: 1, max_retries: 5, delay_seconds: 1, message: 'HTTP 503'}},
    {event_type: 'model_retry_stopped', timestamp: '2026-10-14T14:50:00Z', level: 'error',
      payload: {status: 'failed', message: 'The model is not supported.'}},
  ];
  context.renderAudit();
  const html = node('#audit-list').innerHTML;
  assert.match(html, /Model retry 1\/5 in 1s: HTTP 503/);
  assert.match(html, /<summary>run · Model retry stopped: The model is not supported\.<\/summary>/);
  assert.doesNotMatch(html, /model_waiting/);
});
