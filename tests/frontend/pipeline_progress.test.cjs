const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
const context = vm.createContext({ document: { addEventListener() {} } });
vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
const progressFor = (state) => context.getPipelineProgress(state);

test('Reference Build stays at its node regardless of the old reported percentage', () => {
  for (const progress of [0, 45, 70, 100]) {
    const result = progressFor({ phase: 'reference_building', status: 'running', progress, t5_stage_complete: true });
    assert.equal(result.percent, 20);
    assert.equal(result.stages[1].active, true);
    assert.equal(result.stages[1].done, false);
    assert.equal(result.stages[2].done, false);
  }
});

test('Reference completion advances beyond its node and stops before Task generation', () => {
  const result = progressFor({ phase: 'references_ready', status: 'paused', progress: 50 });
  assert.equal(result.percent, 30);
  assert.equal(result.stages[1].done, true);
  assert.equal(result.stages[2].active, false);
  assert.equal(result.stages[2].done, false);
});

test('all active stages stay at their corresponding nodes', () => {
  for (const [phase, percent, index] of [
    ['normalizing', 10, 0],
    ['reference_building', 20, 1],
    ['discovering_task_trajectories', 40, 2],
    ['inferring_trajectories', 40, 2],
    ['inducing_skills', 70, 3],
    ['validating', 70, 3],
    ['committing', 70, 3],
  ]) {
    const result = progressFor({ phase, status: 'running', progress: 100 });
    assert.equal(result.percent, percent, phase);
    assert.equal(result.stages[index].active, true, phase);
    assert.equal(result.stages[index].done, false, phase);
  }
});

test('completed checkpoints advance between nodes without starting the next stage', () => {
  assert.equal(progressFor({ phase: 'corpus_frozen' }).percent, 15);
  assert.equal(progressFor({ phase: 'tasks_ready', status: 'paused' }).percent, 40);
});

test('stopped runs show completed stages instead of the backend terminal 100 percent', () => {
  for (const phase of ['failed', 'cancelled']) {
    const result = progressFor({ phase, status: phase, progress: 100, t5_stage_complete: true, t6_reference_complete: true });
    assert.equal(result.percent, 30);
    assert.equal(result.stages.some(stage => stage.active), false);
    assert.equal(result.stages[2].done, false);
  }
  assert.equal(progressFor({ phase: 'failed', progress: 100 }).percent, 0);
});

test('completed runs including completion with gaps fill the entire progress bar', () => {
  for (const phase of ['completed', 'degraded']) {
    const result = progressFor({ phase, status: phase });
    assert.equal(result.percent, 100);
    assert.equal(result.stages.every(stage => stage.done), true);
  }
});

test('new runs begin at zero and legacy stage names keep the same mapping', () => {
  assert.equal(progressFor({ phase: 'created', status: 'queued', progress: 15 }).percent, 0);
  assert.equal(progressFor({ stage: 'registered', progress: 15 }).percent, 0);
  assert.equal(progressFor({ stage: 'reference_build', status: 'running' }).percent, 20);
  assert.equal(progressFor({ phase: 'unknown', stage: 'reference_building' }).percent, 20);
  assert.equal(progressFor({}).percent, 0);
});

test('Skill remains the final visible stage until backend finalization completes', () => {
  for (const phase of ['inducing_skills', 'validating', 'committing', 'registry_commit']) {
    const result = progressFor({ phase, status: 'running', t7_skill_complete: true });
    assert.equal(result.stages.length, 4);
    assert.equal(result.percent, 70);
    assert.equal(result.stages[3].active, true);
    assert.equal(result.stages[3].done, false);
    assert.equal(result.stages[3].label, phase === 'inducing_skills' ? 'In progress' : 'Finalizing');
  }
});

test('a finalization failure cannot display the pipeline as complete', () => {
  const result = progressFor({ phase: 'failed', status: 'failed', t5_stage_complete: true, t6_reference_complete: true, t7_trajectory_complete: true, t7_skill_complete: true });
  assert.equal(result.percent, 55);
  assert.equal(result.stages[3].done, false);
});

test('source discovery keeps its extra node and uses the same shortened final stage', () => {
  const waiting = progressFor({ source_mode: 'discovery', phase: 'awaiting_source_selection', status: 'waiting_for_input' });
  assert.equal(waiting.stages.length, 5);
  assert.equal(waiting.percent, 20);
  const finalizing = progressFor({ source_mode: 'discovery', phase: 'committing', status: 'running', t7_skill_complete: true });
  assert.equal(finalizing.stages.length, 5);
  assert.equal(finalizing.percent, 75);
  assert.equal(finalizing.stages[4].active, true);
  assert.equal(finalizing.stages[4].label, 'Finalizing');
  assert.equal(progressFor({ source_mode: 'discovery', phase: 'completed' }).percent, 100);
});

test('discovery uses its five new percentages while manual progress is unchanged', () => {
  const phases = ['normalizing', 'reference_building', 'discovering_task_trajectories', 'inducing_skills'];
  const percentages = (source_mode) => phases.map(phase => progressFor({ source_mode, phase, status: 'running' }).percent);
  assert.deepEqual(percentages('manual'), [10, 20, 40, 70]);
  assert.deepEqual(percentages('discovery'), [25, 35, 50, 75]);
  assert.equal(progressFor({ source_mode: 'discovery', phase: 'discovering_sources', status: 'running' }).percent, 20);
});

test('discovery checkpoint and stopped progress use the new scale without implying completion', () => {
  for (const [phase, percent] of [['created', 0], ['source_selected', 22.5], ['corpus_frozen', 30], ['references_ready', 42.5]]) {
    assert.equal(progressFor({ source_mode: 'discovery', phase }).percent, percent);
  }
  const stopped = progressFor({ source_mode: 'discovery', phase: 'failed', status: 'failed', selected_candidate_id: 'candidate', t5_stage_complete: true, t6_reference_complete: true, t7_trajectory_complete: true });
  assert.equal(stopped.percent, 62.5);
  assert.equal(stopped.stages[4].done, false);
});

test('progress never moves backwards through checkpoint and active phases', () => {
  for (const source_mode of ['manual', 'discovery']) {
    let previous = 0;
    const phases = ['created', ...(source_mode === 'discovery' ? ['discovering_sources', 'awaiting_source_selection', 'source_selected'] : []), 'acquiring', 'normalizing', 'corpus_frozen', 'reference_building', 'references_ready', 'discovering_task_trajectories', 'inducing_skills', 'validating', 'committing', 'completed'];
    for (const phase of phases) {
      const percent = progressFor({ source_mode, phase }).percent;
      assert.ok(percent >= previous, `${source_mode}/${phase}: ${percent} < ${previous}`);
      previous = percent;
    }
  }
});

test('visual fill follows the evenly spaced active stage without changing its numeric progress', () => {
  const cases = [
    ['manual', 'normalizing', 10, 12.5],
    ['manual', 'reference_building', 20, 37.5],
    ['manual', 'discovering_task_trajectories', 40, 62.5],
    ['manual', 'inducing_skills', 70, 87.5],
    ['discovery', 'discovering_sources', 20, 10],
    ['discovery', 'normalizing', 25, 30],
    ['discovery', 'reference_building', 35, 50],
    ['discovery', 'discovering_task_trajectories', 50, 70],
    ['discovery', 'inducing_skills', 75, 90],
  ];
  for (const [source_mode, phase, numeric, visual] of cases) {
    const result = progressFor({ source_mode, phase, status: 'running' });
    assert.equal(result.percent, numeric);
    assert.equal(result.trackPercent, visual);
    assert.ok(result.trackInset > 0 && result.trackInset <= 24);
  }
});

test('visual fill starts empty, ends full only on completion, and stops between finished and pending stages', () => {
  for (const source_mode of ['manual', 'discovery']) {
    const initial = progressFor({ source_mode, phase: 'created', status: 'queued' });
    assert.equal(initial.trackPercent, 0);
    assert.equal(initial.trackInset, 0);
    for (const phase of ['completed', 'degraded']) {
      const finished = progressFor({ source_mode, phase });
      assert.equal(finished.trackPercent, 100);
      assert.equal(finished.trackInset, 0);
    }
  }
  const paused = progressFor({ source_mode: 'discovery', phase: 'references_ready', status: 'paused' });
  assert.equal(paused.percent, 42.5);
  assert.equal(paused.trackPercent, 60);
  assert.equal(paused.trackInset, 0);
  const failed = progressFor({ phase: 'failed', status: 'failed', t5_stage_complete: true, t6_reference_complete: true });
  assert.equal(failed.percent, 30);
  assert.equal(failed.trackPercent, 50);
  assert.equal(failed.trackInset, 0);
});
