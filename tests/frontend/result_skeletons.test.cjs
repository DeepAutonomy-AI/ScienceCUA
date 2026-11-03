const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function renderer() {
  const fields = { '#evidence-filter': { value: '' }, '#evidence-modality': { value: 'all' } };
  for (const selector of ['#tab-task-trajectory', '#task-count', '#trajectory-count', '#task-list', '#trajectory-list']) {
    fields[selector] = {
      attributes: {}, writes: 0, textContent: '', markup: '',
      setAttribute(name, value) { this.attributes[name] = value; },
      set innerHTML(value) { this.markup = value; this.writes++; },
      get innerHTML() { return this.markup; },
    };
  }
  const context = vm.createContext({ document: { addEventListener() {}, querySelector: selector => fields[selector] } });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname, '../../app/static/app.js'), 'utf8'), context);
  const state = vm.runInContext('appState', context);
  const set = (status = 'running', phase = 'normalizing', result = {}, source_mode = 'manual') => {
    state.runPayload = { state: { status, phase, source_mode }, result };
  };
  return { context, state, fields, set };
}

test('pending empty result sections show skeletons while running in either source mode', () => {
  const { context, set } = renderer();
  for (const mode of ['manual', 'discovery']) {
    set('running', 'normalizing', {}, mode);
    for (const section of ['evidence', 'reference', 'tasks', 'trajectories', 'skills']) assert.equal(context.shouldShowResultSkeleton(section), true);
    set('running', 'reference_building', {}, mode);
    assert.equal(context.shouldShowResultSkeleton('evidence'), false);
    assert.equal(context.shouldShowResultSkeleton('reference'), true);
    set('running', 'inducing_skills', {}, mode);
    assert.equal(context.shouldShowResultSkeleton('reference'), false);
    assert.equal(context.shouldShowResultSkeleton('skills'), true);
  }
});

test('paused, stopped, terminal and waiting-for-input runs do not imply ongoing loading', () => {
  const { context, set } = renderer();
  for (const status of ['paused', 'cancelling', 'waiting_for_input', 'completed', 'degraded', 'failed', 'cancelled']) {
    set(status);
    for (const section of ['evidence', 'reference', 'tasks', 'trajectories', 'skills']) assert.equal(context.shouldShowResultSkeleton(section), false, status);
  }
  for (const phase of ['completed', 'degraded', 'failed', 'cancelled', 'unknown', 'awaiting_source_selection']) {
    set('running', phase, {}, 'discovery');
    for (const section of ['tasks', 'trajectories', 'skills']) assert.equal(context.shouldShowResultSkeleton(section), false, phase);
  }
});

test('real results and alternate Evidence representations always take precedence', () => {
  const { context, set } = renderer();
  for (const snapshot of [
    { evidence: [{ evidence_id: 'E1' }] },
    { evidence_trees: [{ nodes: [{ evidence_node_id: 'N1' }] }] },
    { source_results: [{ evidence_tree: { nodes: [{ evidence_node_id: 'N1' }] } }] },
  ]) {
    set('running', 'normalizing', { snapshot });
    assert.equal(context.shouldShowResultSkeleton('evidence'), false);
  }
  set('running', 'reference_building', { references: [{ reference_id: 'R1' }] });
  assert.equal(context.shouldShowResultSkeleton('reference'), false);
  set('running', 'inducing_skills', { skills: [{ title: 'Manual' }] });
  assert.equal(context.shouldShowResultSkeleton('skills'), false);
});

test('search and modality filters keep their true empty state', () => {
  const { context, fields, set } = renderer();
  set();
  fields['#evidence-filter'].value = 'no match';
  assert.equal(context.shouldShowResultSkeleton('evidence'), false);
  fields['#evidence-filter'].value = '';
  fields['#evidence-modality'].value = 'pdf';
  assert.equal(context.shouldShowResultSkeleton('evidence'), false);
});

test('skeleton markup is deterministic, escaped and translated', () => {
  const { context, state } = renderer();
  assert.equal(context.renderResultSkeleton('skills'), context.renderResultSkeleton('skills'));
  assert.match(context.renderResultSkeleton('skills', '<prepare> & wait'), /&lt;prepare&gt; &amp; wait/);
  assert.match(context.renderResultSkeleton('reference'), /aria-hidden="true"/);
  state.language = 'Chinese';
  assert.match(context.renderResultSkeleton('evidence'), /正在准备证据/);
  assert.match(context.renderResultSkeleton('tasks'), /正在准备任务/);
  assert.match(context.renderResultSkeleton('trajectories'), /正在准备轨迹/);
});

test('task and trajectory loading follows their shared stage and completion checkpoint', () => {
  const { context, state, set } = renderer();
  for (const mode of ['manual', 'discovery']) {
    for (const status of ['queued', 'running']) {
      for (const phase of ['acquiring', 'reference_building', 'discovering_task_trajectories', 'trajectory_inference']) {
        set(status, phase, {}, mode);
        for (const section of ['tasks', 'trajectories']) assert.equal(context.shouldShowResultSkeleton(section), true, `${mode}: ${status}: ${phase}: ${section}`);
      }
    }
    set('running', 'trajectory_inference', {}, mode);
    state.runPayload.state.t7_trajectory_complete = true;
    for (const section of ['tasks', 'trajectories']) assert.equal(context.shouldShowResultSkeleton(section), false);
    set('running', 'inducing_skills', {}, mode);
    for (const section of ['tasks', 'trajectories']) assert.equal(context.shouldShowResultSkeleton(section), false);
  }
});

test('task and trajectory columns show partial results independently and reset busy state', () => {
  const { context, fields, set } = renderer();
  set('running', 'discovering_task_trajectories');
  context.renderTasksAndTrajectories();
  assert.match(fields['#task-list'].innerHTML, /Preparing Tasks…/);
  assert.match(fields['#trajectory-list'].innerHTML, /Preparing Trajectories…/);
  assert.equal(fields['#tab-task-trajectory'].attributes['aria-busy'], 'true');
  context.renderTasksAndTrajectories();
  assert.equal(fields['#task-list'].writes, 1, 'Task polling retains skeleton DOM');
  assert.equal(fields['#trajectory-list'].writes, 1, 'Trajectory polling retains skeleton DOM');

  const tasks = [{ task_id: 'T1', task: 'Render a graph', citations: [] }];
  set('running', 'trajectory_inference', { tasks });
  context.renderTasksAndTrajectories();
  assert.match(fields['#task-list'].innerHTML, /Render a graph/);
  assert.doesNotMatch(fields['#task-list'].innerHTML, /result-skeleton/);
  assert.match(fields['#trajectory-list'].innerHTML, /Preparing Trajectories…/);
  assert.equal(fields['#task-list'].attributes['aria-busy'], 'false');
  assert.equal(fields['#trajectory-list'].attributes['aria-busy'], 'true');
  assert.equal(fields['#tab-task-trajectory'].attributes['aria-busy'], 'true');
  assert.equal(fields['#task-count'].textContent, 1);

  set('running', 'trajectory_inference', { trajectories: [{ task_id: 'T1', trajectory_id: 'TR1', steps: [] }] });
  context.renderTasksAndTrajectories();
  assert.match(fields['#task-list'].innerHTML, /Preparing Tasks…/, 'A new pending result can restore the task skeleton');
  assert.doesNotMatch(fields['#trajectory-list'].innerHTML, /result-skeleton/);
  assert.equal(fields['#task-list'].attributes['aria-busy'], 'true');
  assert.equal(fields['#trajectory-list'].attributes['aria-busy'], 'false');

  set('running', 'inducing_skills');
  context.renderTasksAndTrajectories();
  for (const selector of ['#task-list', '#trajectory-list']) {
    assert.match(fields[selector].innerHTML, /No submitted artifacts yet/);
    assert.equal(fields[selector].attributes['aria-busy'], 'false');
  }
  assert.equal(fields['#tab-task-trajectory'].attributes['aria-busy'], 'false');
});

test('unchanged skeleton polling reuses the DOM and a changed result can replace it', () => {
  const { context } = renderer();
  let writes = 0;
  const list = { set innerHTML(value) { this.markup = value; writes++; } };
  const markup = context.renderResultSkeleton('reference');
  context.updateResultListMarkup(list, markup);
  context.updateResultListMarkup(list, markup);
  assert.equal(writes, 1);
  context.updateResultListMarkup(list, 'Actual Reference');
  assert.equal(writes, 2);
  assert.equal(list.markup, 'Actual Reference');
});
