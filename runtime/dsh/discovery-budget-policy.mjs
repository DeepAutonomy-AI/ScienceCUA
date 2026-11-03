import { SEARCH_FAILURE_MARKER } from './search-protocols.mjs'
import { toolDiagnostic } from './tool-diagnostics.mjs'
// Discovery-only limits and durable, model-visible feedback after every tool.
// Semantic manual-root and version decisions remain with the Agent.
export function createDiscoveryBudget(config, now = Date.now) {
  const deadline = Number(config.deadline_epoch_ms)
  const maxRounds = Number(config.max_search_rounds)
  const maxFetches = Number(config.max_fetches_per_round)
  const wrapUp = Number(config.wrap_up_seconds)
  const recoveryCooldown = Number(config.recovery_cooldown_ms ?? 5000)
  const maxRecoveryInspections = Number(config.max_recovery_inspections ?? 3)
  if (!Number.isFinite(deadline) || !Number.isInteger(maxRounds) || maxRounds < 1 || !Number.isInteger(maxFetches) ||
      maxFetches < 1 || !Number.isFinite(wrapUp) || wrapUp < 0 ||
      !Number.isFinite(recoveryCooldown) || recoveryCooldown < 0 || recoveryCooldown > 30000 ||
      !Number.isInteger(maxRecoveryInspections) || maxRecoveryInspections < 0 || maxRecoveryInspections > 6) {
    throw new Error('Invalid discovery budget configuration')
  }
  let searches = 0, fetches = 0, saved = [], broadSearchSeen = false
  let searchFailures = 0, blocked = false
  let cooldownUntil = 0, recoveryAvailable = false, recoveryUsed = false
  let recoveryInspections = 0, failedProbeSeen = false
  let validationPending = 0
  const completedInspections = new Set(), pendingInspections = new Set()
  const pendingSearches = []
  const parsedArgs = args => {
    if (typeof args !== 'string') return args || {}
    try { return JSON.parse(args) } catch { return {} }
  }
  const inspectionKey = value => {
    if (typeof value !== 'string') return null
    const text = value.trim()
    try {
      const url = new URL(text)
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || /\s/.test(text)) return null
      // Match the host's canonical_candidate_url: do not collapse dot segments,
      // default ports or percent encodings that its cache treats separately.
      const match = /^(https?):\/\/([^/?#]+)([^?#]*)(\?[^#]*)?(?:#.*)?$/i.exec(text)
      return match ? `${match[1].toLowerCase()}://${match[2].toLowerCase()}${match[3] || '/'}${match[4]?.length > 1 ? match[4] : ''}` : null
    } catch { return null }
  }
  const inspectionValue = result => {
    if (result.value && typeof result.value === 'object') return result.value
    for (const block of result.content || []) {
      if (block.type !== 'text') continue
      try { const value = JSON.parse(block.text); if (value && typeof value === 'object') return value } catch {}
    }
    return {}
  }
  const queryText = args => {
    if (typeof args === 'string') { try { args = JSON.parse(args) } catch { return '' } }
    return Array.isArray(args?.queries) ? args.queries.join(' ') : String(args?.query || '')
  }
  const targetPattern = config.target_version ? new RegExp(`(^|[^\\w])${String(config.target_version).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?=$|[^\\w])`) : null
  const isBroad = query => !/\b\d+(?:\.\d+)+\b/.test(query) && !(targetPattern?.test(query))
  const remaining = () => Math.max(0, Math.ceil((deadline - now()) / 1000))
  return {
    async waitForRecovery(name, signal) {
      if (name !== 'web_search' || blocked || pendingSearches.length) return
      const delay = Math.max(0, Math.min(cooldownUntil, deadline) - now())
      if (!delay) return
      await new Promise((resolve, reject) => {
        const finish = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
        const abort = () => { finish(); reject(new Error('Discovery recovery wait was cancelled')) }
        const timer = setTimeout(() => { finish(); resolve() }, delay)
        signal?.addEventListener('abort', abort, { once: true })
        if (signal?.aborted) abort()
      })
    },
    before(name, args) {
      if (!['web_search', 'web_fetch', 'inspect_source_candidate'].includes(name)) return
      args = parsedArgs(args)
      if (name === 'inspect_source_candidate' && args?.action === 'skip') return
      if (remaining() === 0) return 'Discovery time budget reached. Finish with the saved candidates.'
      if (name === 'web_search') {
        if (pendingSearches.length) return 'Wait for the current web_search result before starting another search.'
        if (validationPending) return 'Read this round\'s automatic validations with next_source_validation before another search. Save suitable results; an exact HTML match does not waive PDF review.'
        if (blocked) return 'Search backend is blocked after bounded retries. Inspect existing candidates and submit with the observed error; do not repeat web_search.'
        if (cooldownUntil > now()) return 'Search backend is cooling down. The next web_search waits automatically before one recovery probe; inspect existing candidates meanwhile.'
        if (searches + pendingSearches.length >= maxRounds) return 'Search rounds are exhausted. Inspect useful existing HTML/PDF leads and save suitable learning collections, then submit.'
        if (remaining() <= wrapUp && saved.length) return 'Time to finish. Use the saved candidates; do not start another search.'
        const query = queryText(args)
        if (config.target_version && query && !broadSearchSeen && !pendingSearches.some(item => item.broad) && searches + pendingSearches.length === maxRounds - 1 &&
            !saved.some(item => item.version_relation === 'exact') && !isBroad(query)) {
          return 'Reserve this final search for a version-free manual/PDF query. Remove version numbers, keep the software identity and manual/PDF terms, then retry. This rejection did not consume a search round.'
        }
        if (recoveryAvailable) {
          recoveryAvailable = false
          recoveryUsed = true
        }
        pendingSearches.push({ query, broad: Boolean(query && isBroad(query)) })
      } else {
        const key = name === 'inspect_source_candidate' ? inspectionKey(args.url) : null
        // The host caches completed probes (including failures) and redirect
        // aliases. Re-reading those results does not make another HTTP request.
        if (key && completedInspections.has(key)) return
        if (fetches >= maxFetches) {
          if (name !== 'inspect_source_candidate' || !(failedProbeSeen || searchFailures > 0 || blocked) ||
              recoveryInspections >= maxRecoveryInspections) {
            return 'This round has used its verification allowance. Save verified candidates, then finish or use a remaining search round. After an observed failure, only inspect_source_candidate may use the separate bounded recovery allowance.'
          }
          recoveryInspections += 1
        } else fetches += 1
        if (key) pendingInspections.add(key)
      }
    },
    observe(name, result, args) {
      args = parsedArgs(args)
      const observed = inspectionValue(result)
      if (config.parallel_validation && observed.validation_state) {
        validationPending = Number(observed.validation_state.pending ?? Math.max(observed.validation_state.active || 0, observed.validation_state.unreviewed || 0))
      }
      if (name === 'next_source_validation' && observed.url && ['available', 'failed'].includes(observed.status)) {
        const key = inspectionKey(observed.url)
        if (key) completedInspections.add(key)
        failedProbeSeen ||= observed.status === 'failed'
      }
      if (name === 'inspect_source_candidate' && args.action !== 'skip') {
        const key = inspectionKey(args.url)
        if (key && pendingInspections.delete(key) && !result.isError) {
          const value = inspectionValue(result)
          if (value.ok === true && ['available', 'failed'].includes(value.status)) {
            completedInspections.add(key)
            failedProbeSeen ||= value.status === 'failed'
            const finalKey = value.status === 'available' && inspectionKey(value.final_url)
            if (finalKey) completedInspections.add(finalKey)
          }
        }
      }
      if (name === 'web_search') {
        const query = queryText(args)
        const index = pendingSearches.findIndex(item => item.query === query)
        if (index < 0) return
        const [pending] = pendingSearches.splice(index, 1)
        // Guard-denied calls never reserved a round and cannot change the circuit.
        if (!pending) return
        const text = (result.content || []).filter(item => item.type === 'text').map(item => item.text || '').join('\n')
        const failed = result.isError || /^Error:/m.test(text) || text.includes(SEARCH_FAILURE_MARKER)
        if (failed) {
          const diagnostic = toolDiagnostic(name, result)
          // Invalid model arguments never reached a search backend. Permit a
          // corrected call without poisoning the service recovery circuit.
          if (diagnostic?.model_can_fix && diagnostic.origin === 'model_arguments') return
          let failure = {}
          try { failure = JSON.parse(text.split(SEARCH_FAILURE_MARKER)[1]?.split('\n')[0] || '{}') } catch {}
          searchFailures += 1
          blocked = Boolean(diagnostic?.fatal || failure.fatal) || (searchFailures >= 2 && recoveryUsed)
          if (!blocked && searchFailures >= 2) {
            cooldownUntil = now() + recoveryCooldown
            recoveryAvailable = true
          }
        } else {
          if (config.parallel_validation) validationPending = 1 // Host result queue must be read after every successful search.
          if (searches > 0) fetches = 0
          searches += 1
          broadSearchSeen ||= pending.broad
          searchFailures = 0
          cooldownUntil = 0
        }
      }
      if (name === 'save_source_candidate' && !result.isError && result.value?.ok === true) {
        saved = result.value.saved_candidates || saved
      }
    },
    feedback() {
      const seconds = remaining()
      const followups = Math.max(0, maxRounds - searches)
      return `DISCOVERY BUDGET UPDATE (host): ${seconds} seconds remaining. ` +
        `Search rounds used: ${searches}/${maxRounds}; search rounds remaining: ${followups}. ` +
        `Version-free manual search attempted: ${broadSearchSeen}. ` +
        `Failed backend search calls: ${searchFailures}; search backend blocked: ${blocked}. ` +
        `Recovery cooldown: ${Math.max(0, Math.ceil((cooldownUntil - now()) / 1000))} seconds; recovery probe used: ${recoveryUsed}. ` +
        `Verifications remaining this round: ${Math.max(0, maxFetches - fetches)}. ` +
        `Recovery inspections remaining: ${Math.max(0, maxRecoveryInspections - recoveryInspections)} ` +
        `(inspect_source_candidate only, after an observed failure; shared across all rounds). ` +
        `Saved candidates: ${saved.length}. ${JSON.stringify(saved)}\n` +
        (config.parallel_validation ? `Automatic validations pending/review required: ${validationPending}. Use next_source_validation to review one completed document at a time. ` : '') +
        'Prefer suitable public HTML learning collections; retain useful PDF alternatives. ' +
        'Review discovered manual links before spending another search on exact-version details. ' +
        'If no exact manual is saved, use a version-free manual/PDF query before the search allowance runs out. ' +
        'After inspecting HTML, compare reasonable broader/narrower learning scopes before saving; a global sidebar alone is insufficient. ' +
        'Prefer a coherent worked-tutorial collection over an umbrella of documentation types; installation/viewer prerequisites or fewer links alone do not justify broadening it. ' +
        'Once a suitable learning entry or PDF is verified, call save_source_candidate BEFORE further research. ' +
        'Reusing a completed host inspection is free. For 403/404, use observed official directory/navigation links or a public alternative; do not repeatedly guess URL paths. ' +
        (recoveryAvailable ? 'One recovery web_search is available; the host waits for the cooldown automatically. ' : '') +
        (seconds <= wrapUp ? 'Wrap up now; preserve verified manuals with honest version labels and submit.' :
          'Stop early once a strong manual is saved and its version assessment is sufficient.')
    },
  }
}

export function installDiscoveryBudget(ctx, config) {
  const budget = createDiscoveryBudget(config)
  const relevant = new Set(['web_search', 'web_fetch', 'inspect_source_candidate', 'next_source_validation', 'list_discovered_sources', 'save_source_candidate', 'submit_source_candidates'])
  // The pinned SDK evaluates guards synchronously; a Promise is interpreted as
  // a guard denial value. Wait in its asynchronous post-execute hook instead,
  // before the model receives feedback offering the recovery call.
  ctx.tools.guard(exec => budget.before(exec.name, exec.args ?? exec.arguments ?? exec.input))
  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next()
    if (!relevant.has(exec.name) || decision.kind !== 'accept') return decision
    budget.observe(exec.name, result, exec.args ?? exec.arguments ?? exec.input)
    await budget.waitForRecovery(exec.name, exec.signal)
    // Post-execute content is committed by DSH, so the next model step sees
    // a fresh clock without mutating a frozen LLM request or tool output value.
    return { ...decision, content: [
      ...(decision.content ?? result.content), { type: 'text', text: budget.feedback() },
    ] }
  })
}
