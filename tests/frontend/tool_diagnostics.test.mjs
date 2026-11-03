import test from 'node:test'
import assert from 'node:assert/strict'
import { apply, toolDiagnostic, diagnosticError, safeText, DIAGNOSTIC_MARKER } from '../../runtime/dsh/tool-diagnostics.mjs'
import { normalizeSearchResult, searchEvidence } from '../../runtime/dsh/search-result-contract.mjs'
import { createDiscoveryBudget } from '../../runtime/dsh/discovery-budget-policy.mjs'

const errorResult = error => ({ isError: true, error: { message: error.message }, content: [{ type: 'text', text: error.message }] })
test('provider contract requires lossless boolean and reports the exact field', () => {
  assert.throws(() => normalizeSearchResult({ sources: [] }), error => {
    const result = toolDiagnostic('web_search', errorResult(error))
    assert.equal(result.field, 'result.truncated')
    assert.equal(result.expected, 'boolean')
    assert.equal(result.fatal, true)
    assert.equal(result.model_can_fix, false)
    return true
  })
  assert.deepEqual(normalizeSearchResult({ sources: [], truncated: false }), { sources: [], truncated: false })
  for (const value of [{}, { sources: {}, truncated: false }, { sources: [{ url: 42 }], truncated: false },
    { sources: [{ url: 'https://example.org', title: 4 }], truncated: false }]) {
    assert.throws(() => normalizeSearchResult(value), /TOOL_OUTPUT_SCHEMA_ERROR/)
  }
  const result = normalizeSearchResult({ sources: [{ url: 'https://example.org/', title: 'Guide' }], truncated: false, summary: 'Summary' })
  assert.deepEqual(JSON.parse(JSON.stringify(result)), result)
  assert.equal(result.content, 'Summary')
})

test('evidence reaches committed model content, and failed batches discard pending evidence', async () => {
  let hook
  const record = searchEvidence({ on: (_, fn) => { hook = fn } })
  record('manual', { sources: [{ url: 'https://example.org/manual.pdf', snippet: 'Version 1' }], truncated: false })
  const result = await hook({ name: 'web_search' }, { content: [] }, async () => ({ kind: 'accept' }))
  assert.match(result.content[0].text, /DISCOVERY SEARCH EVIDENCE/)
  assert.match(result.content[0].text, /manual.pdf/)
  record('manual', { sources: [], truncated: false })
  await hook({ name: 'web_search' }, { isError: true }, async () => ({ kind: 'accept' }))
  assert.deepEqual(await hook({ name: 'web_search' }, {}, async () => ({ kind: 'accept' })), { kind: 'accept' })
})

function policy() {
  let guard, post
  apply({ tools: { guard: fn => { guard = fn } }, on: (_, fn) => { post = fn } })
  return { guard, post: (exec, result) => post(exec, result, async () => ({ kind: 'accept' })) }
}
test('input correction is allowed; identical repeated failures are blocked without poisoning the backend', async () => {
  const { guard, post } = policy()
  const exec = { name: 'web_search', arguments: {} }
  const result = { isError: true, error: { code: 'INVALID_ARGS', message: 'queries: missing required property' }, content: [] }
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000, max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90 })
  for (let i = 0; i < 2; i++) {
    budget.before('web_search', {})
    budget.observe('web_search', result, {})
    const feedback = await post(exec, result)
    assert.match(feedback.content[0].text, /queries/)
    assert.match(feedback.content[0].text, /"model_can_fix":true/)
  }
  assert.match(guard(exec), /Repeated identical failing call/)
  assert.equal(guard({ name: 'web_search', arguments: { queries: ['manual'] } }), undefined)
  assert.match(budget.feedback(), /Failed backend search calls: 0/)
  const upstream = await post({ name: 'web_search', arguments: { queries: ['rejected'] } }, errorResult(diagnosticError({
    origin: 'model_arguments', code: 'upstream_query_rejected', field: 'query', model_can_fix: true, fatal: false })))
  assert.match(upstream.content[1].text, /"category":"model_arguments"/)
})
test('fatal output errors immediately stop only the broken tool and its search circuit', async () => {
  const { guard, post } = policy()
  const result = errorResult(diagnosticError({ origin: 'server_adapter', code: 'TOOL_OUTPUT_SCHEMA_ERROR', field: 'result.truncated' }))
  const feedback = await post({ name: 'web_search', arguments: { queries: ['one'] } }, result)
  assert.match(feedback.content[0].text, /result.truncated/)
  assert.match(guard({ name: 'web_search', arguments: { queries: ['different'] } }), /TOOL_OUTPUT_SCHEMA_ERROR/)
  assert.equal(guard({ name: 'save_source_candidate' }), undefined)
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000, max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90 })
  budget.before('web_search'); budget.observe('web_search', result)
  assert.match(budget.before('web_search'), /blocked/)
})
test('transient failures are bounded and redaction covers model-visible diagnostics', async () => {
  const { guard, post } = policy()
  const exec = { name: 'web_fetch', arguments: { url: 'https://example.org' } }
  for (let i = 0; i < 3; i++) await post(exec, errorResult(new Error('HTTP 503 upstream unavailable')))
  assert.match(guard(exec), /Repeated identical failing call/)
  const value = toolDiagnostic('web_search', errorResult(new Error('HTTP 401 sk-fake-secret Bearer hidden-token api_key=other-token test-known-key')), ['test-known-key'])
  assert.doesNotMatch(JSON.stringify(value), /fake-secret|hidden-token|other-token|test-known-key/)
  assert.equal(safeText('private test-known-key', ['test-known-key']), 'private [REDACTED]')
  assert.ok(DIAGNOSTIC_MARKER)
})
