const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appPath = path.resolve(__dirname, '../../app/static/app.js');
const source = fs.readFileSync(appPath, 'utf8');

function createApp() {
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', textContent: '', innerHTML: '', hidden: false, disabled: false, open: false,
      style: { setProperty() {} }, classList: { add() {}, remove() {}, toggle() {} },
      querySelector: (child) => child.includes(':checked') ? null : node(`${selector} ${child}`),
      querySelectorAll: () => [], setAttribute() {}, removeAttribute() {}, getAttribute: () => null,
      closest: () => null, contains: () => false, insertAdjacentElement() {}, focus() {}, remove() {},
      showModal() { this.open = true; }, close() { this.open = false; },
      scrollIntoView() {}, addEventListener() {},
    });
    return nodes.get(selector);
  };
  class FakeFormData {
    constructor(form) { this.values = new Map(Object.entries(form?.values || {})); }
    get(key) { return this.values.get(key); }
    set(key, value) { this.values.set(key, value); }
    delete(key) { this.values.delete(key); }
    keys() { return this.values.keys(); }
    append(key, value) { this.values.set(key, value); }
  }
  const requests = [];
  const context = vm.createContext({
    document: { addEventListener() {}, querySelector: node, querySelectorAll: () => [],
      createElement: () => ({ remove() {} }), documentElement: {} },
    window: {}, FormData: FakeFormData, URL, crypto: { randomUUID: () => 'url-1' },
    clearTimeout() {}, setTimeout() {},
    fetch: async (url, options) => { requests.push({ url, options }); return { ok: true, json: async () => ({ run_id: 'run-1' }) }; },
  });
  vm.runInContext(source, context, { filename: appPath });
  vm.runInContext('selectRun = async () => {}; refreshRuns = async () => {}; pollRun = async () => {};', context);
  const state = vm.runInContext('appState', context);
  state.setupStage = 'model';
  state.sourceModeChosen = true;
  node('#run-form input[name=\'target_version\']').value = '1.9';
  return { context, state, node, requests };
}

test('model form initializes from public backend defaults', async () => {
  const { context, node } = createApp();
  context.fetch = async () => ({ ok: true, json: async () => ({
    base_url: 'https://configured.example/v1', model: 'configured-model',
    api_protocol: 'responses', reasoning_effort: 'medium',
  }) });
  await context.loadModelDefaults();
  assert.equal(node("#api-fields [name='api_base_url']").value, 'https://configured.example/v1');
  assert.equal(node("#api-fields [name='api_model']").value, 'configured-model');
  assert.equal(node("#api-fields [name='api_protocol']").value, 'responses');
  assert.equal(node("#api-fields [name='reasoning_effort']").value, 'medium');
});

test('delayed defaults preserve a model and endpoint the user already edited', async () => {
  const { context, state, node } = createApp();
  state.editedModelFields.add('api_base_url');
  state.editedModelFields.add('api_model');
  node("#api-fields [name='api_base_url']").value = 'https://edited.example/responses';
  node("#api-fields [name='api_model']").value = 'edited-model';
  node("#api-fields [name='api_protocol']").value = 'responses';
  context.fetch = async () => ({ ok: true, json: async () => ({
    base_url: 'https://configured.example/v1', model: 'configured-model',
    api_protocol: 'chat_completions', reasoning_effort: 'medium',
  }) });
  await context.loadModelDefaults();
  assert.equal(node("#api-fields [name='api_base_url']").value, 'https://edited.example/responses');
  assert.equal(node("#api-fields [name='api_model']").value, 'edited-model');
  assert.equal(node("#api-fields [name='api_protocol']").value, 'responses');
});

test('manual form enforces one source and does not contact the backend for empty or multiple input', async () => {
  const { context, state, requests, node } = createApp();
  const event = { preventDefault() {}, target: { values: {} } };
  await context.submitRun(event);
  assert.equal(requests.length, 0);
  assert.match(node('#form-error').textContent, /exactly one/);
  state.files = [{ file: { name: 'manual.pdf' }, version: '1.9' }];
  state.urls = [{ url: 'https://example.org/manual/', version: '1.9' }];
  await context.submitRun(event);
  assert.equal(requests.length, 0);
});

test('discovery submits no material and binds output language without requiring a source', async () => {
  const { context, state, requests } = createApp();
  state.sourceMode = 'discovery';
  state.language = 'Chinese';
  // Previously entered manual material must not leak into the discovery path.
  state.files = [{ file: { name: 'old.pdf' }, version: '1.0' }];
  state.urls = [{ url: 'https://example.org/old/', version: '1.0' }];
  await context.submitRun({ preventDefault() {}, target: { values: { software_name: 'ChimeraX', api_base_url: 'https://example.org/v1' } } });
  assert.equal(requests.length, 1);
  const form = requests[0].options.body;
  assert.equal(form.get('source_mode'), 'discovery');
  assert.equal(form.get('output_language'), 'Chinese');
  assert.equal(form.get('urls_json'), '[]');
  assert.equal(form.get('file_versions_json'), '[]');
  assert.equal(form.get('files'), undefined);
  assert.equal(form.get('ui_locale'), undefined);
  assert.equal(form.get('operation_path_preference'), undefined);
});

test('manual form passes the selected file and task language', async () => {
  const { context, state, requests } = createApp();
  const file = { name: 'manual.pdf' };
  state.files = [{ file, version: '1.9' }];
  await context.submitRun({ preventDefault() {}, target: { values: { software_name: 'ChimeraX' } } });
  const form = requests[0].options.body;
  assert.equal(form.get('source_mode'), 'manual');
  assert.equal(form.get('output_language'), 'English');
  assert.equal(form.get('files'), file);
  assert.equal(form.get('operation_path_preference'), undefined);
});

test('material choice only advances after Target Environment is complete', () => {
  const { context, state, node } = createApp();
  state.setupStage = 'target';
  state.sourceModeChosen = false;
  context.continueToSources();
  assert.equal(state.setupStage, 'target');
  assert.equal(node('#model-configuration').open, false);
  node("#run-form [name='software_name']").value = 'ChimeraX';
  node("#run-form [name='target_version']").value = '1.9';
  node("#run-form [name='platform']").value = 'General';
  context.continueToSources();
  assert.equal(state.setupStage, 'sources');
  assert.equal(node('#source-mode-dialog').open, true);
  context.chooseSourceMode('discovery');
  assert.equal(state.sourceMode, 'discovery');
  assert.equal(node('#manual-source-inputs').hidden, true);
  assert.equal(state.setupStage, 'model');
  assert.equal(node('#model-configuration').open, true);
  context.chooseSourceMode('manual');
  assert.equal(node('#manual-source-inputs').hidden, false);
  assert.equal(node('#source-guide').hidden, false);
});

test('discovery progress waits at its search node without starting Source extraction', () => {
  const { context } = createApp();
  const progress = context.getPipelineProgress({ source_mode: 'discovery', phase: 'awaiting_source_selection', status: 'waiting_for_input' });
  assert.equal(progress.stages.length, 5);
  assert.equal(progress.stages[0].active, true);
  assert.equal(progress.stages[0].paused, true);
  assert.equal(progress.stages[0].label, 'Waiting for your selection');
  assert.equal(progress.stages[1].active, false);
  assert.equal(context.getPipelineProgress({ source_mode: 'discovery', phase: 'created', status: 'queued' }).percent, 0);
  const selected = context.getPipelineProgress({ source_mode: 'discovery', phase: 'source_selected', status: 'queued' });
  assert.equal(selected.stages[0].done, true);
  assert.equal(selected.stages[1].active, false);
  const parsing = context.getPipelineProgress({ source_mode: 'discovery', phase: 'normalizing', status: 'running' });
  assert.equal(parsing.stages[0].done, true);
  assert.equal(parsing.stages[1].active, true);
});

test('candidate dialog handles one to three results and escapes their text without translating source claims', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.runPayload = {
    state: { target_profile: { target_version: '1.9' } },
    source_discovery: { candidates: Array.from({ length: 4 }, (_, index) => ({ candidate_id: `c${index}`, title: '<Manual>', url: 'https://example.org/manual/', modality: 'html', source_type: 'official', version_evidence: 'Exact English quote', html_scope: {
      scope_summary: 'Complete <tutorial collection>', included_sections: ['Loading', 'Analysis'],
      excluded_sections: ['Developer API'], limitations: ['Unverified appendix <script>'],
    } })) },
  };
  context.renderCandidateChoices();
  const html = node('#source-candidates-list').innerHTML;
  assert.equal((html.match(/type="radio"/g) || []).length, 3);
  assert.match(html, /&lt;Manual&gt;/);
  assert.match(html, /Exact English quote/);
  assert.match(html, /版本依据/);
  assert.match(html, /资料覆盖范围/);
  assert.match(html, /Complete &lt;tutorial collection&gt;/);
  assert.match(html, /Loading; Analysis/);
  assert.match(html, /Developer API/);
  assert.match(html, /Unverified appendix &lt;script&gt;/);
  assert.doesNotMatch(html, /<script>/);
  assert.match(node('#source-candidates-summary').textContent, /HTML 手册将抓取/);
  assert.equal(context.safeSourceUrl('javascript:alert(1)'), '');
  assert.equal(context.safeSourceUrl('https://example.org/manual/'), 'https://example.org/manual/');
});

for (const modality of ['pdf', 'html']) {
  test(`selected ${modality} material keeps its original link visible in the Sources table`, () => {
    const { context, state, node } = createApp();
    const url = `https://example.org/manual${modality === 'pdf' ? '.pdf' : '/'}?chapter=1&view=full`;
    state.runPayload = { result: { snapshot: { source_results: [{
      source_id: 'SRC1', display_name: '<Original manual>', modality,
      manifest: { source_origin: 'websearch', source_priority: 'user_selected',
        origin_url: url, final_url: 'https://cdn.example.org/redirected-material', resolved_version: '1.9' },
    }] } } };
    context.renderSources();
    const html = node('#source-table-body').innerHTML;
    const escapedUrl = url.replace(/&/g, '&amp;');
    assert.ok(html.includes(`href="${escapedUrl}"`));
    assert.ok(html.includes(`>${escapedUrl}</a>`));
    assert.match(html, /target="_blank" rel="noopener noreferrer"/);
    assert.match(html, /&lt;Original manual&gt;/);
    assert.doesNotMatch(html, /<Original manual>|cdn\.example\.org/);
  });
}

test('source links fall back to the final URL without linking local uploads or unsafe schemes', () => {
  const { context, state, node } = createApp();
  state.runPayload = { result: { snapshot: { source_results: [
    { source_id: 'SRC1', display_name: 'Redirected manual', modality: 'html',
      manifest: { origin_url: 'javascript:alert(1)', final_url: 'https://example.org/manual/' } },
    { source_id: 'SRC2', display_name: 'Local PDF', modality: 'pdf', manifest: {} },
    { source_id: 'SRC3', display_name: 'Local HTML', modality: 'html',
      manifest: { origin_url: 'file:///C:/manual.html', final_url: 'data:text/html,unsafe' } },
  ] } } };
  context.renderSources();
  const html = node('#source-table-body').innerHTML;
  assert.equal((html.match(/class="source-origin-link"/g) || []).length, 1);
  assert.match(html, /href="https:\/\/example\.org\/manual\/"/);
  assert.match(html, /Local PDF|Local HTML/);
  assert.doesNotMatch(html, /javascript:|file:\/\/\/|data:text/);
});

test('selected discovery material is visible during acquisition even with an older backend and no result', () => {
  const { context, state, node } = createApp();
  state.runPayload = {
    state: { phase: 'acquiring', status: 'running', selected_candidate_id: 'chosen' }, result: null,
    source_discovery: { candidates: [
      { candidate_id: 'other', title: 'Unselected manual', url: 'https://example.org/other', modality: 'pdf' },
      { candidate_id: 'chosen', title: '<GRASS manual>', url: 'https://example.org/manual?x=1&y=2',
        document_version: '8.4.3dev', modality: 'html', version_relation: 'newer' },
    ] },
  };
  const before = JSON.stringify(state.runPayload);
  context.renderSources();
  const html = node('#source-table-body').innerHTML;
  assert.equal(node('#source-waiting').hidden, true);
  assert.equal((html.match(/<tr>/g) || []).length, 1);
  assert.match(html, /&lt;GRASS manual&gt;/);
  assert.match(html, /href="https:\/\/example.org\/manual\?x=1&amp;y=2"/);
  assert.match(html, /8\.4\.3dev|Newer version/);
  assert.match(html, /Acquiring source material/);
  assert.doesNotMatch(html, /Unselected manual|undefined|<GRASS manual>/);
  assert.equal(JSON.stringify(state.runPayload), before);
});

test('unselected search candidates do not appear as adopted sources', () => {
  const { context, state, node } = createApp();
  state.runPayload = { state: { phase: 'awaiting_source_selection' }, result: null,
    source_discovery: { candidates: [{ candidate_id: 'unselected', title: 'Candidate' }] } };
  context.renderSources();
  assert.equal(node('#source-waiting').hidden, false);
  assert.equal(node('#source-table-body').innerHTML, '');
});

test('source previews show processing and interrupted states before final results replace them', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.runPayload = { state: { phase: 'source_review', status: 'running' }, result: null, sources: [{
    source_id: 'SRC1', display_name: 'Provisional title', modality: 'pdf', is_preview: true,
    manifest: { resolved_version: '2.9' },
  }] };
  context.renderSources();
  assert.match(node('#source-table-body').innerHTML, /正在处理资料/);
  state.runPayload.state.status = 'cancelled';
  context.renderSources();
  assert.match(node('#source-table-body').innerHTML, /已停止/);
  assert.doesNotMatch(node('#source-table-body').innerHTML, /正在处理资料/);
  state.runPayload.state = { phase: 'inducing_skills', status: 'running' };
  state.runPayload.result = { snapshot: { source_results: [{
    source_id: 'SRC1', display_name: 'Final title', modality: 'pdf', status: 'succeeded',
    manifest: { resolved_version: '2.9' },
  }] } };
  context.renderSources();
  const html = node('#source-table-body').innerHTML;
  assert.equal((html.match(/<tr>/g) || []).length, 1);
  assert.match(html, /Final title/);
  assert.doesNotMatch(html, /Provisional title|正在处理资料|资料已选中|已停止/);
});

test('an excluded source keeps its own status instead of the running phase', () => {
  const { context, state, node } = createApp();
  state.runPayload = { state: { phase: 'acquiring', status: 'running' }, sources: [{
    display_name: 'Unsupported manual', modality: 'unknown', status: 'excluded', is_preview: true, manifest: {},
  }] };
  context.renderSources();
  assert.match(node('#source-table-body').innerHTML, /excluded/);
  assert.doesNotMatch(node('#source-table-body').innerHTML, /Acquiring source material/);
});

test('waiting dialog can be dismissed and re-opened without popping up on every poll', () => {
  const { context, state, node } = createApp();
  state.currentRunId = 'waiting-1';
  state.runPayload = { state: { phase: 'awaiting_source_selection', status: 'waiting_for_input' }, source_discovery: { candidates: [] } };
  context.renderSourceSelectionState(state.runPayload.state);
  assert.equal(node('#source-candidates-dialog').open, true);
  assert.equal(node('#source-candidates-continue').hidden, true);
  node('#source-candidates-dialog').close();
  context.renderSourceSelectionState(state.runPayload.state);
  assert.equal(node('#source-candidates-dialog').open, false);
  context.showSourceCandidates();
  assert.equal(node('#source-candidates-dialog').open, true);
});

test('selection sends exactly one candidate with the current model key and prevents duplicate submission', async () => {
  const { context, state, node } = createApp();
  state.currentRunId = state.candidatesRunId = 'waiting-1';
  state.runPayload = { state: { requires_api_key: false } };
  node('#run-form input[name=\'api_key\']').value = 'memory-only-key';
  let release;
  const requests = [];
  context.fetch = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    await new Promise((resolve) => { release = resolve; });
    return { ok: true, json: async () => ({ status: 'queued' }) };
  };
  const first = context.submitDiscoveryAction('select-source', { candidate_id: 'c1' });
  await context.submitDiscoveryAction('select-source', { candidate_id: 'c1' });
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], { url: '/api/runs/waiting-1/select-source', body: { candidate_id: 'c1', api_key: 'memory-only-key' } });
  release();
  await first;
  assert.equal(state.candidateRequestPending, false);
});

test('restored run requires a key before continuation and provides a translated actionable error', async () => {
  const { context, state, node, requests } = createApp();
  state.language = 'Chinese';
  state.currentRunId = state.candidatesRunId = 'waiting-1';
  state.runPayload = { state: { requires_api_key: true } };
  await context.submitDiscoveryAction('search-again', {});
  assert.equal(requests.length, 0);
  assert.match(node('#source-candidates-error').textContent, /API Key/);
  assert.match(node('#source-candidates-error').textContent, /填写/);
  assert.equal(node('#model-configuration').open, true);
});

test('translation changes UI labels while preserving reference and skill source text', () => {
  const { context, state } = createApp();
  state.language = 'Chinese';
  assert.equal(context.t('Choose material'), '选择资料');
  assert.equal(context.stageName('discovering_sources'), '搜索手册中');
  assert.equal(context.statusLabel('waiting_for_input'), '等待你选择资料');
  assert.equal(context.t('{count} citations', { count: 3 }), '3 个引用');
  assert.match(context.renderSkillPreconditions('Keep this English source text.'), /Keep this English source text\./);
});


test('deadline candidate list remains selectable with a localized explanation', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.runPayload = { state: { target_profile: { target_version: '1.9' } }, source_discovery: {
    completion_reason: 'time_budget_reached', candidates: [{ candidate_id: 'candidate_saved',
      title: 'Saved manual', url: 'https://example.org/manual/', modality: 'html', version_relation: 'unknown' }],
  } };
  context.renderCandidateChoices();
  assert.equal(node('#source-candidates-notice').hidden, false);
  assert.notEqual(node('#source-candidates-notice').textContent, '');
  assert.equal(node('#source-candidates-continue').hidden, false);
  assert.match(node('#source-candidates-list').innerHTML, /candidate_saved/);
});

test('search service failures are distinguished from completed searches with no matching manuals', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.currentRunId = 'waiting-search';
  state.runPayload = { state: { phase: 'awaiting_source_selection', status: 'waiting_for_input' },
    source_discovery: { candidates: [], search_status: 'unavailable' } };
  context.renderSourceSelectionState(state.runPayload.state);
  assert.match(node('#source-selection-notice-text').textContent, /搜索服务失败/);
  assert.match(node('#source-candidates-summary').textContent, /搜索服务失败/);
  assert.doesNotMatch(node('#source-candidates-summary').textContent, /未找到可用手册/);
  state.runPayload.source_discovery.search_status = 'completed';
  context.renderCandidateChoices();
  assert.match(node('#source-candidates-summary').textContent, /未找到可用手册/);
});

test('empty discovery explains host-classified failures in English and Chinese with actionable alternatives', () => {
  const cases = [
    ['search_unsupported', /verifiable web search/, /可核实的联网搜索/],
    ['search_authentication', /credentials or access permissions/, /凭据或访问权限/],
    ['search_quota', /no available quota/, /额度不足/],
    ['search_rate_limit', /rate limited/, /请求频率限制/],
    ['search_configuration', /configuration could not be used/, /搜索配置无法使用/],
    ['search_unavailable', /search service failed/, /搜索服务失败/],
    ['source_authentication', /some inspected.*sign-in or authorization/, /部分文档页面要求登录或授权/],
    ['source_access_denied', /some inspected.*refused access/, /部分文档页面拒绝访问/],
    ['source_not_found', /some inspected.*404/, /部分文档链接返回 404/],
    ['source_unavailable', /documentation could not be retrieved/, /未能获取所检查的文档内容/],
    ['no_suitable_sources', /no usable manual was found within this search/, /本次搜索未找到可用手册/],
  ];
  for (const language of ['English', 'Chinese']) {
    const { context, state, node } = createApp();
    state.language = language;
    for (const [reason, english, chinese] of cases) {
      state.runPayload = { state: {}, source_discovery: {
        candidates: [], search_status: 'completed', failure_summary: { primary_reason: reason },
        summary: '<script>arbitrary model claim</script>', search_notes: ['https://secret.example/?api_key=private'],
        search_error: { category: 'unavailable', message: 'Bearer private-key' },
      } };
      context.renderCandidateChoices();
      const message = node('#source-candidates-summary').textContent;
      assert.match(message, language === 'English' ? english : chinese, `${language}: ${reason}`);
      assert.match(message, language === 'English' ? /Sorry/ : /很抱歉/);
      assert.match(message, language === 'English' ? /upload a PDF or HTML manual/ : /自行上传 PDF 或 HTML 手册/);
      assert.match(message, language === 'English' ? /official manual link/ : /官方手册链接/);
      assert.doesNotMatch(message, /script|arbitrary|private|Bearer|secret\.example/);
    }
  }
});

test('historical search errors retain specific recovery advice without exposing raw errors', () => {
  const { context, state, node } = createApp();
  for (const [category, expected] of [
    ['authentication', /credentials or access permissions/], ['quota', /no available quota/],
    ['rate_limit', /rate limited/], ['configuration', /configuration could not be used/],
    ['unsupported', /verifiable web search/], ['protocol', /verifiable web search/],
    ['unverified_retrieval', /verifiable web search/], ['upstream', /search service failed/],
  ]) {
    state.runPayload = { state: {}, source_discovery: { candidates: [], search_status: 'unavailable',
      search_error: { category, message: 'Do not show: api_key=private-value' } } };
    context.renderCandidateChoices();
    assert.match(node('#source-candidates-summary').textContent, expected);
    assert.doesNotMatch(node('#source-candidates-summary').textContent, /private-value|api_key/);
  }
});

test('unrecognized failure classifications are ignored and the waiting notice matches the dialog', () => {
  const { context, state, node } = createApp();
  state.currentRunId = 'unknown-failure';
  state.runPayload = { state: { phase: 'awaiting_source_selection', status: 'waiting_for_input' },
    source_discovery: { candidates: [], search_status: 'completed',
      failure_summary: { primary_reason: '__proto__' },
      search_error: { category: 'constructor', message: 'untrusted detail' }, summary: 'model explanation' } };
  context.renderSourceSelectionState(state.runPayload.state);
  assert.match(node('#source-selection-notice-text').textContent, /Sorry, no usable manual was found/);
  assert.equal(node('#source-selection-notice-text').textContent, node('#source-candidates-summary').textContent);
  assert.doesNotMatch(node('#source-selection-notice-text').textContent, /untrusted|model explanation|__proto__/);
});

test('mixed accessible and failed material inspections retain the primary result and explain partial access failures', () => {
  const { context, state, node } = createApp();
  for (const language of ['English', 'Chinese']) {
    state.language = language;
    state.runPayload = { state: {}, source_discovery: { candidates: [], search_status: 'completed',
      failure_summary: { primary_reason: 'no_suitable_sources', inspection_counts: { available: 1, failed: 2 } },
      summary: 'Do not display this unverified model explanation.', search_notes: ['All official documentation requires a private key.'],
    } };
    context.renderCandidateChoices();
    const message = node('#source-candidates-summary').textContent;
    assert.match(message, language === 'English' ? /Sorry, no usable manual was found/ : /很抱歉，本次搜索未找到可用手册/);
    assert.match(message, language === 'English' ? /Some inspected material links could not be accessed\./ : /部分已检查的资料链接无法访问。/);
    assert.match(message, language === 'English' ? /upload a PDF or HTML manual/ : /自行上传 PDF 或 HTML 手册/);
    assert.doesNotMatch(message, /unverified|private key|All official/);
    for (const failed of [0, undefined, -1, '2']) {
      state.runPayload.source_discovery.failure_summary.inspection_counts.failed = failed;
      context.renderCandidateChoices();
      assert.doesNotMatch(node('#source-candidates-summary').textContent, /Some inspected material links|部分已检查/);
    }
  }
});

test('partial search failures retain selectable saved manuals and an explanatory notice', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  state.runPayload = { state: {}, source_discovery: { search_status: 'partial', candidates: [
    { candidate_id: 'saved_manual', title: 'Verified manual', url: 'https://manual.example/guide.pdf', modality: 'pdf' },
  ] } };
  context.renderCandidateChoices();
  assert.match(node('#source-candidates-notice').textContent, /部分搜索失败/);
  assert.equal(node('#source-candidates-notice').hidden, false);
  assert.equal(node('#source-candidates-continue').hidden, false);
  assert.match(node('#source-candidates-list').innerHTML, /saved_manual/);
});

test('interrupted and timed-out searches do not claim that no matching manual exists', () => {
  const { context, state, node } = createApp();
  state.language = 'Chinese';
  for (const reason of ['time_budget_reached', 'runtime_interrupted', 'session_ended']) {
    state.runPayload = { state: {}, source_discovery: { candidates: [], completion_reason: reason } };
    context.renderCandidateChoices();
    assert.doesNotMatch(node('#source-candidates-summary').textContent, /未找到可用手册/);
    assert.match(node('#source-candidates-summary').textContent, /时间已到|搜索中断/);
  }
});
