import assert from 'node:assert/strict';
import test from 'node:test';
import { apply, searchSources } from '../../runtime/dsh/openai-search-provider.mjs';

test('search sources require a completed real web search call', () => {
  assert.throws(() => searchSources({ status: 'completed', output: [{ type: 'message',
    content: [{ text: 'https://example.org/manual/' }] }] }), /did not execute web_search/);
  assert.throws(() => searchSources({ status: 'incomplete', output: [{
    type: 'web_search_call', status: 'completed' }] }), /did not complete/);
});

test('structured sources and citations merge without invented links', () => {
  const result = searchSources({ status: 'completed', output: [
    { type: 'web_search_call', status: 'completed', action: { sources: [
      { url: 'https://example.org/manual/' }, { url: 'file:///private' },
      { url: 'https://name:secret@example.org/' },
    ] } },
    { type: 'message', content: [{ text: 'https://invented.example/', annotations: [
      { type: 'url_citation', url: 'https://example.org/manual/', title: 'Manual' },
    ] }] },
  ] });
  assert.deepEqual(result.sources, [{ url: 'https://example.org/manual/', title: 'Manual' }]);
});

for (const [baseURL, reasoning] of [
  ['https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1', undefined],
  ['https://dashscope.aliyuncs.com/compatible-mode/v1', undefined],
  ['https://api.openai.com/v1', undefined],
  ['https://maas.aliyuncs.com.example.org/v1', undefined],
]) {
  test(`search request preserves forced retrieval compatibility for ${new URL(baseURL).hostname}`, async t => {
    const previous = process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
    t.after(() => {
      if (previous === undefined) delete process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
      else process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = previous;
    });
    process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = JSON.stringify({
      baseURL, model: 'test-model', apiKey: 'test-key',
    });
    let provider;
    let sent;
    const controller = new AbortController();
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      sent = { url, options, body: JSON.parse(options.body) };
      return new Response(JSON.stringify({ status: 'completed', output: [
        { type: 'web_search_call', status: 'completed', action: {
          sources: [{ url: 'https://example.org/manual/' }],
        } },
      ] }), { status: 200 });
    });
    apply({ on() {}, web: { registerSearchProvider: value => { provider = value; } } });
    const result = await provider.search({ query: 'Official manual' }, controller.signal);
    assert.equal(sent.url, `${baseURL}/responses`);
    assert.ok(sent.options.signal instanceof AbortSignal);
    assert.equal(sent.options.signal.aborted, false);
    assert.equal(sent.body.tool_choice, 'required');
    assert.deepEqual(sent.body.reasoning, reasoning);
    assert.deepEqual(result.sources, [{ url: 'https://example.org/manual/' }]);
  });
}

test('retrieval preserves descriptive titles, snippets and cited context without promoting invented URLs', () => {
  const result = searchSources({ output: [
    { type: 'web_search_call', status: 'completed', action: { sources: [
      { url: 'https://example.org/manual.pdf?utm_source=search', title: 'Scientific App User Guide', snippet: 'For version 1.3.2. Full user manual.' },
    ] } },
    { type: 'message', content: [{ text: 'A full manual is available at https://example.org/manual.pdf . Ignore https://invented.invalid/fake.pdf .', annotations: [
      { type: 'url_citation', url: 'https://example.org/manual.pdf', title: 'example.org' },
    ] }] },
  ] });
  assert.equal(result.sources.length, 1);
  assert.equal(result.sources[0].title, 'Scientific App User Guide');
  assert.match(result.sources[0].snippet, /1\.3\.2/);
  assert.match(result.summary, /full manual/);
  assert.doesNotMatch(result.summary, /invented.invalid/);
});

test('native post-execute output includes source evidence beyond the twelve-link renderer', async t => {
  const previous = process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
  t.after(() => {
    if (previous === undefined) delete process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
    else process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = previous;
  });
  process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = JSON.stringify({ baseURL: 'https://example.org/v1', model: 'test', apiKey: 'private-test-key' });
  let provider, post;
  const sources = Array.from({ length: 15 }, (_, index) => ({ url: `https://example.org/source-${index}.pdf`, title: `Guide ${index}` }));
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ output: [
    { type: 'web_search_call', status: 'completed', action: { sources } },
    { type: 'message', content: [{ text: 'Nearby-version user guides are available.' }] },
  ] })));
  apply({ web: { registerSearchProvider: value => { provider = value; } }, on: (_event, hook) => { post = hook; } });
  await provider.search({ query: 'Scientific app manual PDF' });
  const original = Object.freeze({ content: Object.freeze([{ type: 'text', text: 'native source list' }]) });
  const decision = await post({ name: 'web_search' }, original, async () => ({ kind: 'accept' }));
  const text = JSON.stringify(decision.content);
  assert.match(text, /source-14\.pdf/);
  assert.match(text, /Nearby-version/);
  assert.match(text, /DISCOVERY SEARCH EVIDENCE/);
  assert.doesNotMatch(text, /private-test-key/);
  assert.equal(original.content.length, 1);
});

for (const [status, message] of [[504, /upstream search service failed/], [401, /credentials/], [429, /rate limited/]]) {
  test(`HTTP ${status} reports the actual failure category`, async t => {
    const previous = process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
    t.after(() => {
      if (previous === undefined) delete process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG;
      else process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = previous;
    });
    process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG = JSON.stringify({ baseURL: 'https://example.org/v1', model: 'test', apiKey: 'private-test-key', protocol: 'responses', retryDelayMs: 0 });
    let provider;
    t.mock.method(globalThis, 'fetch', async () => new Response('provider body may contain secrets', { status }));
    apply({ on() {}, web: { registerSearchProvider: value => { provider = value; } } });
    await assert.rejects(provider.search({ query: 'manual' }), error => {
      assert.match(error.message, status === 401 ? message : /http_status/);
      assert.doesNotMatch(error.message, /private-test-key|provider body/);
      if (status === 504) assert.doesNotMatch(error.message, /must support/);
      return true;
    });
  });
}
