import assert from 'node:assert/strict'
import test from 'node:test'
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createSearchClient } from '../../runtime/dsh/search-request.mjs'
import { createTraceFetch } from '../../runtime/dsh/http-trace.mjs'
import { createDiscoveryBudget } from '../../runtime/dsh/discovery-budget-policy.mjs'
import { SEARCH_FAILURE_MARKER, messagesSearchSources, chatSearchSources, searchEndpoint } from '../../runtime/dsh/search-protocols.mjs'

const config = { baseURL: 'https://gateway.example/v1', model: 'arbitrary-model', apiKey: 'search-private-key', retryDelayMs: 0 }
const sources = [{ url: 'https://manual.example/guide.pdf', title: 'Complete user manual' }]
const response = value => new Response(JSON.stringify(value), { headers: { 'content-type': 'application/json' } })
const retrieved = () => response({ output: [{ type: 'web_search_call', status: 'completed', action: { sources } }] })
const messages = () => response({ content: [
  { type: 'server_tool_use', id: 'search1', name: 'web_search' },
  { type: 'web_search_tool_result', tool_use_id: 'search1', content: sources.map(item => ({ type: 'web_search_result', ...item })) },
  { type: 'text', text: 'A complete user guide is available.' },
] })

test('503 on Responses recovers through Messages without changing the model and caches the working protocol', async () => {
  const calls = []
  const client = createSearchClient(config, async (url, options) => {
    const body = JSON.parse(options.body)
    calls.push({ url, body, headers: options.headers })
    assert.equal(body.model, 'arbitrary-model')
    return url.endsWith('/responses') ? new Response('{"error":{"code":"route_unavailable","message":"No available route"}}', { status: 503 }) : messages()
  })
  const first = await client.search('Scientific software user manual')
  assert.deepEqual(first.sources, sources)
  assert.equal(first.protocol, 'messages')
  assert.equal(first.recovered_from.length, 3)
  assert.equal(calls.length, 4)
  assert.equal(calls[3].headers['x-api-key'], config.apiKey)
  assert.equal(calls[3].body.tools[0].type, 'web_search_20250305')
  await client.search('Another manual query')
  assert.equal(calls.length, 5)
  assert.ok(calls[4].url.endsWith('/messages'))
})

test('Chat search is negotiated using structured citations when both tool APIs are unavailable', async () => {
  const urls = []
  const client = createSearchClient(config, async (url, options) => {
    urls.push(url)
    if (!url.endsWith('/chat/completions')) return new Response('{"error":{"message":"Not supported"}}', { status: 404 })
    const body = JSON.parse(options.body)
    assert.deepEqual(body.web_search_options, {})
    assert.equal(body.tools, undefined)
    return response({ choices: [{ message: { content: 'Retrieved guide', annotations: [
      { type: 'url_citation', url_citation: sources[0] },
    ] } }] })
  })
  assert.deepEqual((await client.search('manual')).sources, sources)
  assert.equal(urls.length, 3)
})

test('a temporary 503 is retried on the same route before negotiating another protocol', async () => {
  let calls = 0
  const client = createSearchClient(config, async () => ++calls === 1 ?
    new Response('{"error":{"code":"temporarily_unavailable"}}', { status: 503 }) : retrieved())
  const result = await client.search('manual')
  assert.equal(result.protocol, 'responses')
  assert.equal(calls, 2)
})

test('unsupported forced search reasoning is adapted from the error, independently of endpoint hostname', async () => {
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'responses' }, async (_url, options) => {
    const body = JSON.parse(options.body)
    if (++calls === 1) return new Response('{"error":{"message":"tool_choice required is incompatible with thinking"}}', { status: 400 })
    assert.deepEqual(body.reasoning, { effort: 'none' })
    return retrieved()
  })
  assert.equal((await client.search('manual')).sources.length, 1)
  assert.equal(calls, 2)
})

test('unsupported optional parameters are removed but actual search execution is still required', async () => {
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'responses' }, async (_url, options) => {
    if (++calls === 1) return new Response('{"error":{"param":"include","message":"Unsupported parameter include"}}', { status: 400 })
    const body = JSON.parse(options.body)
    assert.equal(body.include, undefined)
    assert.equal(body.tool_choice, undefined)
    return retrieved()
  })
  assert.equal((await client.search('manual')).protocol, 'responses')
})

test('plain text, Markdown URLs and client function requests never become retrieved search sources', async () => {
  let calls = 0
  const client = createSearchClient(config, async () => {
    calls++
    return response({ output: [{ type: 'message', content: [{ text: 'https://fake.example/' }] }],
      content: [{ type: 'text', text: '[Fake](https://fake.example/)' }],
      choices: [{ message: { content: 'https://fake.example/', tool_calls: [{ type: 'function', function: { name: 'web_search' } }] } }] })
  })
  await assert.rejects(client.search('manual'), /"category":"unavailable"/)
  await assert.rejects(client.search('different query'), /"category":"unavailable"/)
  assert.equal(calls, 6, 'Unverified answers are rejected without permanently disabling the protocols')
})

test('HTTP error bodies are consumed for backend trace and public errors redact credentials', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'search-error-trace-'))
  t.after(() => { assert.ok(directory.startsWith(join(tmpdir(), 'search-error-trace-'))); rmSync(directory, { recursive: true, force: true }) })
  const fetcher = createTraceFetch({ directory, max_response_bytes: 65536, max_session_bytes: 65536,
    secrets: [config.apiKey] }, async () => new Response(JSON.stringify({ error: {
      code: 'unsupported_model', type: 'routing_error', message: `Model route unavailable. Echo ${config.apiKey}`,
    } }), { status: 503, headers: { 'x-request-id': 'request-debug-1' } }))
  const client = createSearchClient({ ...config, protocol: 'responses' }, fetcher)
  await assert.rejects(client.search('manual'), error => {
    assert.match(error.message, /unsupported_model|routing_error/)
    assert.match(error.message, /request-debug-1/)
    assert.doesNotMatch(error.message, /search-private-key/)
    return true
  })
  const metaFile = readdirSync(directory).find(name => name.endsWith('.meta.json'))
  const meta = JSON.parse(readFileSync(join(directory, metaFile), 'utf8'))
  assert.equal(meta.complete, true)
  assert.ok(meta.captured_bytes > 0)
  const body = readFileSync(join(directory, meta.response_body_file), 'utf8')
  assert.match(body, /unsupported_model/)
  assert.doesNotMatch(body, /search-private-key/)
})

test('authentication failures on an explicit protocol stop without retrying', async () => {
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'responses' }, async () => {
    calls++
    return new Response('{"error":{"message":"Unauthorized"}}', { status: 401 })
  })
  await assert.rejects(client.search('manual'), /"category":"authentication"/)
  await assert.rejects(client.search('manual'), /"fatal":true/)
  assert.equal(calls, 1)
})

test('automatic negotiation can recover from protocol-specific authentication on the same origin', async () => {
  const client = createSearchClient(config, async url => {
    assert.equal(new URL(url).origin, new URL(config.baseURL).origin)
    return url.endsWith('/messages') ? messages() : new Response('{"error":{"message":"This route requires another authentication format"}}', { status: 401 })
  })
  assert.equal((await client.search('manual')).protocol, 'messages')
})

test('cancellation during backoff stops all retry and fallback requests', async () => {
  const controller = new AbortController()
  let calls = 0
  const client = createSearchClient({ ...config, retryDelayMs: 1000 }, async () => {
    calls++
    setTimeout(() => controller.abort(), 10)
    return new Response('{"error":{"message":"Service unavailable"}}', { status: 503 })
  })
  await assert.rejects(client.search('manual', controller.signal), /cancelled/)
  assert.equal(calls, 1)
})

test('explicit endpoint paths are normalized and explicit protocol disables automatic fallback', async () => {
  assert.equal(searchEndpoint('https://gateway.example/compatible/v1/responses/', 'messages'), 'https://gateway.example/compatible/v1/messages')
  assert.equal(searchEndpoint('https://gateway.example/v1/chat/completions', 'responses'), 'https://gateway.example/v1/responses')
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'messages' }, async url => {
    calls++
    assert.ok(url.endsWith('/messages'))
    return new Response('{}', { status: 404 })
  })
  await assert.rejects(client.search('manual'))
  assert.equal(calls, 1)
})

test('Messages tool failures and Chat without annotations are not successful empty searches', () => {
  assert.throws(() => messagesSearchSources({ content: [{ type: 'web_search_tool_result',
    content: { type: 'web_search_tool_result_error', error_code: 'unavailable' } }] }), /unavailable/)
  assert.throws(() => chatSearchSources({ choices: [{ message: { content: 'No sources' } }] }), /no structured/)
})

test('failed searches do not consume successful rounds, renew fetch allowances or satisfy the broad query requirement', () => {
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000,
    max_search_rounds: 3, max_fetches_per_round: 2, wrap_up_seconds: 90, target_version: '1.7.0' })
  budget.before('web_fetch'); budget.before('web_fetch')
  const broad = { queries: ['Scientific manual PDF'] }
  budget.before('web_search', broad)
  budget.observe('web_search', { isError: true, content: [{ type: 'text', text: SEARCH_FAILURE_MARKER +
    JSON.stringify({ category: 'unavailable', retryable: true, fatal: false }) }] }, broad)
  assert.match(budget.feedback(), /Search rounds used: 0\/3/)
  assert.match(budget.feedback(), /Version-free manual search attempted: false/)
  assert.match(budget.before('web_fetch'), /verification allowance/)
  assert.equal(budget.before('web_search', broad), undefined)
  budget.observe('web_search', { content: [] }, broad)
  assert.match(budget.feedback(), /Search rounds used: 1\/3/)
  assert.match(budget.feedback(), /Version-free manual search attempted: true/)
})

test('two failed backend calls cool down without consuming any successful search round', () => {
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000,
    max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90 })
  for (let index = 0; index < 2; index++) {
    assert.equal(budget.before('web_search'), undefined)
    budget.observe('web_search', { isError: true, content: [] })
  }
  assert.match(budget.before('web_search'), /cooling down/)
  assert.match(budget.feedback(), /Search rounds used: 0\/3/)
  assert.equal(budget.before('submit_source_candidates'), undefined)
})
