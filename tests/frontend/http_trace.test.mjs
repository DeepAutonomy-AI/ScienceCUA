import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { createTraceFetch } from '../../runtime/dsh/http-trace.mjs'

function setup(t, overrides = {}) {
  const directory = mkdtempSync(join(tmpdir(), 'http-trace-test-'))
  t.after(() => {
    assert.ok(directory.startsWith(join(tmpdir(), 'http-trace-test-')))
    rmSync(directory, { recursive: true, force: true })
  })
  return { directory, run_id: 'run_test', session_id: 'child_test', stage: 'source_discovery',
    max_response_bytes: 1024 * 1024, max_session_bytes: 4 * 1024 * 1024,
    secrets: ['model-secret', 'search-secret'], ...overrides }
}

function records(config) {
  return readdirSync(config.directory).filter(name => name.endsWith('.meta.json')).sort().map(name => {
    const meta = JSON.parse(readFileSync(join(config.directory, name), 'utf8'))
    return { meta, body: readFileSync(join(config.directory, meta.response_body_file), 'utf8'),
      request: JSON.parse(readFileSync(join(config.directory, name.replace('.meta.json', '.request.json')), 'utf8')) }
  })
}

test('raw SSE reaches the SDK unchanged and credentials split across chunks are redacted on disk', async t => {
  const config = setup(t)
  const wire = Buffer.from('data: {"id":"call_1","arguments":"{\\"queries\\":[\\"手册\\"]}","echo":"model-secret search-secret"}\n\ndata: [DONE]\n\n')
  const fetcher = createTraceFetch(config, async () => new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < wire.length; i += 3) controller.enqueue(wire.subarray(i, i + 3))
      controller.close()
    },
  }), { headers: { 'content-type': 'text/event-stream', 'x-request-id': 'request_1', 'set-cookie': 'private' } }))
  const result = await fetcher('https://model.example/v1/chat/completions?api_key=model-secret', {
    method: 'POST', headers: { authorization: 'Bearer model-secret' }, body: JSON.stringify({
      model: 'test', stream: true, messages: [{ role: 'user', content: 'private prompt' }],
      tools: [{ type: 'function', function: { name: 'web_search', parameters: {
        type: 'object', required: ['queries'], properties: { queries: { type: 'array' } },
      } } }],
    }),
  })
  assert.deepEqual(Buffer.from(await result.arrayBuffer()), wire)
  const [saved] = records(config)
  assert.equal(saved.body, wire.toString().replaceAll('model-secret', '[REDACTED]').replaceAll('search-secret', '[REDACTED]'))
  assert.equal(saved.meta.complete, true)
  assert.equal(saved.meta.truncated, false)
  assert.equal(saved.meta.received_bytes, wire.length)
  assert.equal(saved.meta.response_body_sha256, createHash('sha256').update(wire).digest('hex'))
  assert.deepEqual(saved.request.tools[0].function.parameters.required, ['queries'])
  assert.equal(saved.meta.response_headers['x-request-id'], 'request_1')
  const all = readdirSync(config.directory).map(name => readFileSync(join(config.directory, name), 'utf8')).join('')
  for (const secret of ['model-secret', 'search-secret', 'private prompt', 'set-cookie']) assert.ok(!all.includes(secret))
})

test('JSON HTTP errors and Request inputs retain status, body, and tool definitions', async t => {
  const config = setup(t)
  const fetcher = createTraceFetch(config, async () => new Response('{"error":"rate limited"}', {
    status: 429, headers: { 'content-type': 'application/json' },
  }))
  const result = await fetcher(new Request('https://model.example/v1/responses', {
    method: 'POST', headers: { 'x-api-key': 'different-secret' },
    body: JSON.stringify({ model: 'test', tools: [{ name: 'web_search' }] }),
  }))
  assert.equal(result.status, 429)
  assert.deepEqual(await result.json(), { error: 'rate limited' })
  const [saved] = records(config)
  assert.equal(saved.meta.status, 429)
  assert.equal(saved.body, '{"error":"rate limited"}')
  assert.deepEqual(saved.request.tools, [{ name: 'web_search' }])
})

test('response and session limits bound log bytes without truncating SDK responses', async t => {
  const config = setup(t, { max_response_bytes: 8, max_session_bytes: 12 })
  const fetcher = createTraceFetch(config, async () => new Response('0123456789abcdef'))
  for (let i = 0; i < 3; i++) {
    assert.equal(await (await fetcher('https://model.example/v1/messages', { method: 'POST' })).text(), '0123456789abcdef')
  }
  const saved = records(config)
  assert.deepEqual(saved.map(item => item.meta.captured_bytes), [8, 4, 0])
  assert.ok(saved.every(item => item.meta.complete && item.meta.truncated))
})

test('cancellation is marked incomplete and keeps bytes already received', async t => {
  const config = setup(t)
  const fetcher = createTraceFetch(config, async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(Buffer.from('first packet')) },
  })))
  const result = await fetcher('https://model.example/v1/responses', { method: 'POST' })
  const reader = result.body.getReader()
  await reader.read()
  await reader.cancel('stopped')
  const [saved] = records(config)
  assert.equal(saved.meta.state, 'cancelled')
  assert.equal(saved.meta.complete, false)
  assert.equal(saved.body, 'first packet')
})

test('network and disk failures do not replace the original fetch behavior', async t => {
  const config = setup(t)
  const error = new Error('failed with model-secret')
  const failed = createTraceFetch(config, async () => { throw error })
  await assert.rejects(failed('https://model.example/v1/chat/completions', { method: 'POST' }), value => value === error)
  const [saved] = records(config)
  assert.equal(saved.meta.state, 'network_error')
  assert.ok(!saved.meta.error.includes('model-secret'))
  const file = join(config.directory, 'not-a-directory')
  writeFileSync(file, 'file')
  const original = new Response('still works')
  const fetcher = createTraceFetch({ ...config, directory: file }, async () => original)
  assert.equal(await fetcher('https://model.example/v1/responses', { method: 'POST' }), original)
  assert.equal(await original.text(), 'still works')
})

test('documentation and capability requests are not intercepted', async t => {
  const config = setup(t)
  const original = new Response('page')
  const fetcher = createTraceFetch(config, async () => original)
  assert.equal(await fetcher('https://docs.example/manual.html'), original)
  assert.equal(await fetcher('http://localhost/capabilities/invoke', { method: 'POST' }), original)
  assert.deepEqual(readdirSync(config.directory), [])
})
