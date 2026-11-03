const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
const source = fs.readFileSync(appPath, 'utf8');

function createApp(status, count) {
  const events = Array.from({ length: count }, (_, index) => ({
    event_id: `evt_${index}`,
    event_type: index === count - 1 ? 'run_completed' : 'model_response_streaming',
  }));
  const timers = [];
  const offsets = [];
  const context = vm.createContext({
    document: { addEventListener() {} },
    clearTimeout() {},
    setTimeout(callback) { timers.push(callback); return timers.length; },
    async fetch(url) {
      if (!url.includes('/events?')) return { ok: true, json: async () => ({ state: { status } }) };
      const after = Number(url.match(/after=(\d+)/)[1]);
      offsets.push(after);
      const page = events.slice(after, after + 500);
      return { ok: true, json: async () => ({ events: page, next_offset: after + page.length }) };
    },
  });
  vm.runInContext(source, context, { filename: appPath });
  context.renderRun = () => {};
  context.showRunFailureIfNeeded = () => {};
  context.refreshRuns = async () => {};
  const state = vm.runInContext('appState', context);
  state.currentRunId = 'run_history';
  return { context, state, timers, offsets };
}

test('finished and paused runs load every audit page, including the final event', async () => {
  for (const status of ['completed', 'degraded', 'failed', 'cancelled', 'paused']) {
    const { context, state, timers, offsets } = createApp(status, 1326);
    await context.pollRun();
    for (let page = 0; timers.length && page < 5; page++) await timers.shift()();
    assert.deepEqual(offsets, [0, 500, 1000], status);
    assert.equal(state.events.length, 1326, status);
    assert.equal(state.eventOffset, 1326, status);
    assert.equal(state.events.at(-1).event_type, 'run_completed', status);
    assert.equal(new Set(state.events.map(event => event.event_id)).size, 1326, status);
    assert.equal(timers.length, 0, status);
  }
});

test('an exact full page checks for the end and then stops polling', async () => {
  const { context, state, timers, offsets } = createApp('completed', 500);
  await context.pollRun();
  assert.equal(timers.length, 1);
  await timers.shift()();
  assert.deepEqual(offsets, [0, 500]);
  assert.equal(state.events.length, 500);
  assert.equal(timers.length, 0);
});

test('a small finished audit stops immediately while a running one keeps polling', async () => {
  for (const status of ['completed', 'running']) {
    const { context, state, timers } = createApp(status, 12);
    await context.pollRun();
    assert.equal(state.events.length, 12);
    assert.equal(timers.length, status === 'running' ? 1 : 0);
  }
});

test('switching runs discards an outstanding audit page', async () => {
  const { context, state, timers } = createApp('completed', 852);
  const originalFetch = context.fetch;
  context.fetch = async url => {
    const response = await originalFetch(url);
    if (url.includes('/events?')) {
      response.json = async () => {
        state.currentRunId = 'run_other';
        state.events = [];
        state.eventOffset = 0;
        return { events: [{ event_id: 'evt_old' }], next_offset: 500 };
      };
    }
    return response;
  };
  await context.pollRun();
  assert.equal(state.currentRunId, 'run_other');
  assert.equal(state.events.length, 0);
  assert.equal(state.eventOffset, 0);
  assert.equal(timers.length, 0);
});
