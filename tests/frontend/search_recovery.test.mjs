import assert from 'node:assert/strict'
import test from 'node:test'
import { createSearchClient } from '../../runtime/dsh/search-request.mjs'
import { SEARCH_FAILURE_MARKER } from '../../runtime/dsh/search-protocols.mjs'

const config = { baseURL: 'https://gateway.example', model: 'arbitrary-model', apiKey: 'test-secret', retryDelayMs: 0 }
const retrieved = () => Response.json({ output: [{ type: 'web_search_call', status: 'completed',
  action: { sources: [{ url: 'https://manual.example/guide.pdf' }] } }] })
const refused = () => Response.json({ error: { type: 'invalid_request_error', message: '请求被拒绝' } }, { status: 400 })
const failure = error => JSON.parse(error.message.split(SEARCH_FAILURE_MARKER)[1])

test('the recorded generic 400 can recover on the same protocol with an unchanged request', async () => {
  const requests = []
  const client = createSearchClient(config, async (url, options) => {
    requests.push({ url, body: options.body })
    return requests.length < 3 ? refused() : retrieved()
  })
  const result = await client.search('Lean 4 4.14.0 official manual')
  assert.equal(result.protocol, 'responses')
  assert.equal(requests.length, 3)
  assert.deepEqual(requests[0], requests[1])
  assert.deepEqual(requests[1], requests[2])
  assert.equal(result.recovered_from.length, 2)
  assert.equal(result.recovered_from[0].category, 'request_rejected')
  assert.equal(result.recovered_from[1].attempt, 2)
})

test('a successful protocol survives a refused query and is retried on the next query', async () => {
  const queries = []
  const client = createSearchClient({ ...config, protocol: 'responses' }, async (_url, options) => {
    const query = JSON.parse(options.body).input
    queries.push(query)
    return query.includes('refused query') ? refused() : retrieved()
  })
  await client.search('first manual')
  await assert.rejects(client.search('refused query'), error => {
    const detail = failure(error)
    assert.equal(detail.fatal, false)
    assert.equal(detail.retryable, true)
    assert.equal(detail.attempts.length, 3)
    return true
  })
  assert.equal((await client.search('next manual')).protocol, 'responses')
  assert.equal(queries.length, 5)
})

test('mixed generic refusals, a missing route and unverified text do not poison the session', async () => {
  let recovered = false
  const urls = []
  const client = createSearchClient(config, async url => {
    urls.push(url)
    if (url.endsWith('/responses')) return recovered ? retrieved() : refused()
    if (url.endsWith('/messages')) return new Response('', { status: 404 })
    return Response.json({ choices: [{ message: { content: 'No live web search; https://guessed.example/' } }] })
  })
  await assert.rejects(client.search('first manual'), error => {
    assert.equal(failure(error).fatal, false)
    assert.equal(failure(error).attempts.length, 5)
    return true
  })
  recovered = true
  assert.equal((await client.search('next manual')).protocol, 'responses')
  assert.equal(urls.length, 6)
})

test('a text-only answer is rejected but does not invalidate later genuine retrieval', async () => {
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'responses' }, async () => ++calls === 1
    ? Response.json({ output: [{ type: 'message', content: [{ text: 'https://guessed.example/' }] }] })
    : retrieved())
  await assert.rejects(client.search('first manual'), error => failure(error).fatal === false)
  assert.equal((await client.search('next manual')).sources[0].url, 'https://manual.example/guide.pdf')
})

test('completed structured searches with zero sources remain valid empty searches', async () => {
  const client = createSearchClient(config, async () => Response.json({ output: [
    { type: 'web_search_call', status: 'completed', action: { sources: [] } },
  ] }))
  assert.deepEqual((await client.search('rare manual')).sources, [])
})

test('explicitly unsupported parameters are not blindly retried and may be cached', async () => {
  let calls = 0
  const client = createSearchClient({ ...config, protocol: 'responses' }, async () => {
    calls++
    return Response.json({ error: { param: 'max_output_tokens', message: 'Unsupported parameter max_output_tokens' } }, { status: 400 })
  })
  await assert.rejects(client.search('manual'), error => failure(error).fatal === true)
  await assert.rejects(client.search('another manual'))
  assert.equal(calls, 1)
})

test('all transiently failing protocols stop after at most nine HTTP attempts per search', async () => {
  let calls = 0
  const client = createSearchClient(config, async () => {
    calls++
    return new Response('Service unavailable', { status: 503 })
  })
  await assert.rejects(client.search('manual'), error => {
    assert.equal(failure(error).attempts.length, 9)
    assert.equal(failure(error).fatal, false)
    return true
  })
  assert.equal(calls, 9)
})

test('an invalid query does not mark the protocol unsupported', async () => {
  const client = createSearchClient({ ...config, protocol: 'responses' }, async (_url, options) =>
    JSON.parse(options.body).input.includes('invalid query')
      ? Response.json({ error: { param: 'input', message: 'Invalid parameter input' } }, { status: 400 })
      : retrieved())
  await assert.rejects(client.search('invalid query'), error => failure(error).fatal === false)
  assert.equal((await client.search('valid manual')).protocol, 'responses')
})

test('a per-search timeout aborts the HTTP request and allows a later search to recover', async () => {
  let calls = 0, aborted = false
  const client = createSearchClient({ ...config, protocol: 'responses', timeoutMs: 20 }, async (_url, { signal }) => {
    if (++calls > 1) return retrieved()
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => {
      aborted = true
      reject(new Error('HTTP request aborted'))
    }, { once: true }))
  })
  await assert.rejects(client.search('manual'), error => {
    assert.equal(failure(error).category, 'timeout')
    assert.equal(failure(error).fatal, false)
    return true
  })
  assert.equal(aborted, true)
  assert.equal((await client.search('next manual')).protocol, 'responses')
})

test('quota errors stop all retries and automatic protocol fallback', async () => {
  let calls = 0
  const client = createSearchClient(config, async () => {
    calls++
    return Response.json({ error: { code: 'insufficient_quota' } }, { status: 429 })
  })
  await assert.rejects(client.search('manual'), error => failure(error).category === 'quota')
  await assert.rejects(client.search('next manual'))
  assert.equal(calls, 1)
})

test('caller cancellation aborts an in-flight request without trying any fallback', async () => {
  const controller = new AbortController()
  let calls = 0, httpSignal
  const client = createSearchClient(config, async (_url, { signal }) => {
    calls++
    httpSignal = signal
    return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('HTTP request aborted')), { once: true })
      controller.abort()
    })
  })
  await assert.rejects(client.search('manual', controller.signal), /cancelled/)
  assert.equal(httpSignal.aborted, true)
  assert.equal(calls, 1)
})

test('a server error envelope with HTTP 200 still rejects unsupported retrieval and caches that route', async () => {
  const urls = []
  const client = createSearchClient(config, async url => {
    urls.push(url)
    if (url.endsWith('/responses')) return Response.json({ error: {
      code: 'unsupported_tool', message: 'web_search is not enabled for this model' } })
    if (url.endsWith('/messages')) return new Response('', { status: 404 })
    return Response.json({ choices: [{ message: { content: 'No live retrieval; https://guessed.example/' } }] })
  })
  for (const query of ['first manual', 'another manual']) {
    await assert.rejects(client.search(query), error => {
      assert.equal(failure(error).reason, 'unverified_retrieval')
      assert.equal(failure(error).fatal, false)
      return true
    })
  }
  assert.equal(urls.filter(url => url.endsWith('/responses')).length, 1)
  assert.equal(urls.filter(url => url.endsWith('/messages')).length, 1)
  assert.equal(urls.filter(url => url.endsWith('/chat/completions')).length, 2)
})

test('explicit search tool unavailability is fatal only after all protocols have proved unsupported', async () => {
  let calls = 0
  const client = createSearchClient(config, async url => {
    calls++
    return url.endsWith('/messages') ? Response.json({ content: [{ type: 'web_search_tool_result',
      content: { type: 'web_search_tool_result_error', error_code: 'web_search_disabled' } }] }) :
      new Response('', { status: 404 })
  })
  for (const query of ['first manual', 'another manual']) {
    await assert.rejects(client.search(query), error => {
      const detail = failure(error)
      assert.equal(detail.reason, 'capability_unavailable')
      assert.equal(detail.category, 'unsupported')
      assert.equal(detail.fatal, true)
      return true
    })
  }
  assert.equal(calls, 3)
})

test('HTTP 200 gateway quota errors are not misreported as unverified model text', async () => {
  let calls = 0
  const client = createSearchClient(config, async () => {
    calls++
    return Response.json({ error: { code: 'insufficient_quota', message: 'No quota available' } })
  })
  await assert.rejects(client.search('manual'), error => failure(error).category === 'quota')
  assert.equal(calls, 1)
})

test('completed Messages retrieval with zero results remains a successful empty search', async () => {
  const client = createSearchClient({ ...config, protocol: 'messages' }, async () => Response.json({
    content: [{ type: 'web_search_tool_result', content: [] }],
  }))
  assert.deepEqual((await client.search('rare manual')).sources, [])
})

test('rate limits retain their actionable category after bounded retries and other missing routes', async () => {
  const client = createSearchClient(config, async url => url.endsWith('/responses')
    ? Response.json({ error: { code: 'rate_limit_exceeded', message: 'Slow down' } }, { status: 429 })
    : new Response('', { status: 404 }))
  await assert.rejects(client.search('manual'), error => {
    const detail = failure(error)
    assert.equal(detail.category, 'rate_limit')
    assert.equal(detail.retryable, true)
    assert.equal(detail.fatal, false)
    return true
  })
})

test('structured Messages rate limits remain distinct from unsupported web search', async () => {
  const client = createSearchClient({ ...config, protocol: 'messages' }, async () => Response.json({
    content: [{ type: 'web_search_tool_result', content: {
      type: 'web_search_tool_result_error', error_code: 'too_many_requests' } }],
  }))
  await assert.rejects(client.search('manual'), error => failure(error).category === 'rate_limit')
})
