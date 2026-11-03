import assert from 'node:assert/strict'
import test from 'node:test'
import { createDiscoveryBudget, installDiscoveryBudget } from '../../runtime/dsh/discovery-budget-policy.mjs'
import { SEARCH_FAILURE_MARKER } from '../../runtime/dsh/search-protocols.mjs'

function setup() {
  let now = 100000
  const budget = createDiscoveryBudget({ deadline_epoch_ms: now + 600000,
    max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90 }, () => now)
  return { budget, advance: ms => { now += ms } }
}

test('automatic validations must be read before the next search, without consuming another network allowance', () => {
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000,
    max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90, parallel_validation: true })
  budget.before('web_search', { queries: ['manual'] })
  budget.observe('web_search', { content: [] }, { queries: ['manual'] })
  assert.match(budget.before('web_search', { queries: ['more'] }), /next_source_validation/)
  const observe = state => budget.observe('next_source_validation', { value: { ok: true, url: 'https://example.org/manual.pdf',
    status: 'available', validation_state: state } })
  observe({ active: 1, unreviewed: 1, pending: 1 })
  assert.match(budget.before('web_search'), /automatic validations/)
  observe({ active: 0, unreviewed: 0, pending: 0 })
  assert.equal(budget.before('web_search', { queries: ['more'] }), undefined)
  assert.match(budget.feedback(), /Verifications remaining this round: 6/)
})

test('one initial search plus exactly two follow-ups, each with bounded verification', () => {
  const { budget } = setup()
  for (let round = 0; round < 3; round++) {
    assert.equal(budget.before('web_search'), undefined)
    budget.observe('web_search', { content: [] })
    for (let fetch = 0; fetch < 6; fetch++) assert.equal(budget.before('web_fetch'), undefined)
    assert.match(budget.before('web_fetch'), /verification allowance/)
  }
  assert.match(budget.before('web_search'), /Search rounds are exhausted/)
  assert.equal(budget.before('save_source_candidate'), undefined)
  assert.equal(budget.before('submit_source_candidates'), undefined)
})

test('fresh time and saved manuals are fed back, wrap-up denies exploration but permits saving', () => {
  const { budget, advance } = setup()
  assert.match(budget.feedback(), /600 seconds remaining/)
  budget.before('web_search')
  budget.observe('web_search', { content: [] })
  budget.observe('save_source_candidate', { value: { ok: true, saved_candidates: [
    { url: 'https://example.org/manual/', version_relation: 'unknown' },
  ] } })
  advance(520000)
  assert.match(budget.feedback(), /80 seconds remaining/)
  assert.match(budget.feedback(), /Saved candidates: 1/)
  assert.match(budget.feedback(), /search rounds remaining: 2/)
  assert.match(budget.before('web_search'), /Time to finish/)
  assert.equal(budget.before('save_source_candidate'), undefined)
  advance(80000)
  assert.match(budget.before('web_fetch'), /time budget reached/)
})

test('last search is reserved for a version-free query when no exact manual is saved', () => {
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000,
    max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90, target_version: '1.7.0' });
  assert.equal(budget.before('web_search', { queries: ['ScientificApp 1.7.0 manual'] }), undefined);
  budget.observe('web_search', { content: [] }, { queries: ['ScientificApp 1.7.0 manual'] });
  assert.equal(budget.before('web_search', { queries: ['ScientificApp 1.7.0 official guide'] }), undefined);
  budget.observe('web_search', { content: [] }, { queries: ['ScientificApp 1.7.0 official guide'] });
  assert.match(budget.before('web_search', { queries: ['ScientificApp 1.7.0 pdf'] }), /version-free/);
  assert.match(budget.feedback(), /Search rounds used: 2\/3/);
  assert.equal(budget.before('web_search', { queries: ['ScientificApp user guide filetype:pdf'] }), undefined);
  budget.observe('web_search', { content: [] }, { queries: ['ScientificApp user guide filetype:pdf'] });
  assert.match(budget.feedback(), /Version-free manual search attempted: true/);
});

test('an early generic query leaves the last round available for a discovered filename', () => {
  const budget = createDiscoveryBudget({ deadline_epoch_ms: Date.now() + 600000,
    max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90, target_version: '1.7.0' });
  budget.before('web_search', { queries: ['ScientificApp 1.7.0 manual'] });
  budget.observe('web_search', { content: [] }, { queries: ['ScientificApp 1.7.0 manual'] });
  budget.before('web_search', { queries: ['ScientificApp user guide PDF'] });
  budget.observe('web_search', { content: [] }, { queries: ['ScientificApp user guide PDF'] });
  assert.equal(budget.before('web_search', { queries: ['ScientificAppGuide-1.6.1.pdf mirror'] }), undefined);
});

test('pre-search fetches cannot evade the first round allowance', () => {
  const { budget } = setup()
  for (let i = 0; i < 6; i++) budget.before('web_fetch')
  assert.equal(budget.before('web_search'), undefined)
  assert.match(budget.before('web_fetch'), /verification allowance/)
})

test('post-execute feedback preserves results and only applies to discovery tools', async () => {
  let guard, post
  installDiscoveryBudget({ tools: { guard: fn => { guard = fn } }, on: (name, fn) => { post = fn } },
    { deadline_epoch_ms: Date.now() + 600000, max_search_rounds: 3, max_fetches_per_round: 6, wrap_up_seconds: 90 })
  assert.equal(guard({ name: 'web_search' }), undefined)
  const result = Object.freeze({ content: Object.freeze([{ type: 'text', text: 'External source text' }]) })
  const decision = await post({ name: 'web_search' }, result, async () => ({ kind: 'accept' }))
  assert.equal(decision.content[0].text, 'External source text')
  assert.match(decision.content[1].text, /DISCOVERY BUDGET UPDATE/)
  assert.equal(result.content.length, 1)
  const unrelated = await post({ name: 'inspect_source' }, result, async () => ({ kind: 'accept' }))
  assert.equal(unrelated.content, undefined)
})

test('two failed searches allow one recovery probe after cooldown, then resume normal rounds on success', () => {
  const { budget, advance } = setup()
  for (let i = 0; i < 2; i++) {
    budget.before('web_search', { query: 'manual' })
    budget.observe('web_search', { isError: true, content: [] }, { query: 'manual' })
  }
  assert.match(budget.before('web_search'), /cooling down/)
  assert.match(budget.feedback(), /search backend blocked: false/)
  advance(5000)
  assert.equal(budget.before('web_search', { query: 'manual' }), undefined)
  budget.observe('web_search', { content: [] }, { query: 'manual' })
  assert.match(budget.feedback(), /Search rounds used: 1\/3/)
  assert.match(budget.feedback(), /Failed backend search calls: 0/)
  assert.equal(budget.before('web_search'), undefined)
})

test('a failed recovery probe ends retries and preserves saved candidates', () => {
  const { budget, advance } = setup()
  budget.observe('save_source_candidate', { value: { ok: true, saved_candidates: [
    { url: 'https://manual.example/guide.pdf', version_relation: 'unknown' },
  ] } })
  for (let i = 0; i < 3; i++) {
    if (i === 2) advance(5000)
    assert.equal(budget.before('web_search'), undefined)
    budget.observe('web_search', { isError: true, content: [] })
  }
  assert.match(budget.before('web_search'), /backend is blocked/)
  assert.match(budget.feedback(), /Saved candidates: 1/)
  assert.match(budget.feedback(), /Search rounds used: 0\/3/)
  assert.equal(budget.before('submit_source_candidates'), undefined)
})

test('fatal configuration errors stop immediately without a recovery probe', () => {
  const { budget, advance } = setup()
  budget.before('web_search')
  budget.observe('web_search', { isError: true, content: [{ type: 'text', text: SEARCH_FAILURE_MARKER +
    JSON.stringify({ category: 'authentication', fatal: true }) }] })
  advance(30000)
  assert.match(budget.before('web_search'), /backend is blocked/)
})

test('the stage deadline prevents a recovery probe even after cooldown', () => {
  const { budget, advance } = setup()
  for (let i = 0; i < 2; i++) {
    budget.before('web_search')
    budget.observe('web_search', { isError: true, content: [] })
  }
  advance(600000)
  assert.match(budget.before('web_search'), /time budget reached/)
})

test('the SDK guard stays synchronous while post-execute waits before offering recovery', async () => {
  let guard, post
  installDiscoveryBudget({ tools: { guard: fn => { guard = fn } }, on: (_name, fn) => { post = fn } },
    { deadline_epoch_ms: Date.now() + 600000, max_search_rounds: 3, max_fetches_per_round: 6,
      wrap_up_seconds: 90, recovery_cooldown_ms: 15 })
  for (let i = 0; i < 2; i++) {
    assert.equal(guard({ name: 'web_search' }), undefined)
    await post({ name: 'web_search' }, { isError: true, content: [] }, async () => ({ kind: 'accept' }))
  }
  assert.equal(guard({ name: 'web_search' }), undefined)
  const decision = await post({ name: 'web_search' }, { content: [] }, async () => ({ kind: 'accept' }))
  assert.match(decision.content.at(-1).text, /recovery probe used: true/)
})

test('failed searches preserve a small inspection-only recovery allowance after fetches are exhausted', () => {
  const { budget } = setup()
  for (let index = 0; index < 6; index++) assert.equal(budget.before('web_fetch'), undefined)
  assert.match(budget.before('inspect_source_candidate', { url: 'https://manual.example/' }), /verification allowance/)
  budget.before('web_search')
  budget.observe('web_search', { isError: true, content: [] })
  for (let index = 0; index < 3; index++) {
    assert.equal(budget.before('inspect_source_candidate', { url: `https://manual.example/${index}` }), undefined)
    assert.match(budget.before('web_fetch'), /verification allowance/)
  }
  assert.match(budget.before('inspect_source_candidate', { url: 'https://manual.example/fourth' }), /verification allowance/)
  assert.match(budget.feedback(), /Recovery inspections remaining: 0/)
})

test('a completed failed probe activates recovery without requiring the search backend to fail', () => {
  const { budget } = setup()
  const args = { url: 'https://manual.example/broken' }
  budget.before('inspect_source_candidate', args)
  budget.observe('inspect_source_candidate', { value: { ok: true, status: 'failed', http_status: 404 } }, args)
  for (let index = 0; index < 5; index++) budget.before('web_fetch')
  assert.equal(budget.before('inspect_source_candidate', { url: 'https://manual.example/real-entry' }), undefined)
  assert.match(budget.feedback(), /Recovery inspections remaining: 2/)
})

test('completed host inspections and their observed redirect aliases can be read without spending budget again', () => {
  const { budget } = setup()
  const args = { url: 'https://manual.example/old#chapter' }
  budget.before('inspect_source_candidate', args)
  budget.observe('inspect_source_candidate', { content: [{ type: 'text', text: JSON.stringify({
    ok: true, status: 'available', final_url: 'https://manual.example/current/' }) }] }, args)
  for (let index = 0; index < 5; index++) budget.before('web_fetch')
  assert.equal(budget.before('inspect_source_candidate', { url: 'https://manual.example/old#another' }), undefined)
  assert.equal(budget.before('inspect_source_candidate', { url: 'https://manual.example/current/' }), undefined)
  assert.match(budget.feedback(), /Verifications remaining this round: 0/)
  assert.match(budget.feedback(), /Recovery inspections remaining: 3/)
})

test('skipped, rejected or unfinished inspections cannot grant free requests or activate recovery', () => {
  const { budget } = setup()
  const args = { url: 'https://manual.example/not-inspected' }
  budget.before('inspect_source_candidate', { ...args, action: 'skip' })
  budget.observe('inspect_source_candidate', { value: { ok: true, status: 'skipped' } }, { ...args, action: 'skip' })
  budget.before('inspect_source_candidate', args)
  budget.observe('inspect_source_candidate', { value: { ok: false, error: 'invalid_request' } }, args)
  for (let index = 0; index < 5; index++) budget.before('web_fetch')
  assert.match(budget.before('inspect_source_candidate', args), /verification allowance/)
})

test('free cache reuse does not normalize URLs more broadly than the Python inspection cache', () => {
  const { budget } = setup()
  const args = { url: 'https://manual.example:443/a/../guide' }
  budget.before('inspect_source_candidate', args)
  budget.observe('inspect_source_candidate', { value: { ok: true, status: 'available' } }, args)
  for (let index = 0; index < 5; index++) budget.before('web_fetch')
  assert.equal(budget.before('inspect_source_candidate', args), undefined)
  assert.match(budget.before('inspect_source_candidate', { url: 'https://manual.example/guide' }), /verification allowance/)
})

test('recovery inspection allowance is shared across all rounds and respects the deadline', () => {
  const { budget, advance } = setup()
  budget.before('web_search')
  budget.observe('web_search', { isError: true, content: [] })
  for (let index = 0; index < 6; index++) budget.before('web_fetch')
  budget.before('inspect_source_candidate', { url: 'https://manual.example/recovery' })
  budget.before('web_search')
  budget.observe('web_search', { content: [] })
  assert.match(budget.feedback(), /Recovery inspections remaining: 2/)
  advance(600000)
  assert.match(budget.before('inspect_source_candidate', { url: 'https://manual.example/late' }), /time budget reached/)
})
