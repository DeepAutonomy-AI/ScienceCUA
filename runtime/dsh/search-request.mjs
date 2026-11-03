import { SEARCH_PROTOCOLS, SEARCH_FAILURE_MARKER, SearchFailure, searchRequest, parseSearchResult } from './search-protocols.mjs'

function redact(value, secret) {
  let text = String(value ?? '')
  for (const token of [secret, encodeURIComponent(secret), JSON.stringify(secret).slice(1, -1)]) {
    if (token) text = text.split(token).join('[REDACTED]')
  }
  return text.replace(/\bBearer\s+[^\s"']+/gi, 'Bearer [REDACTED]')
    .replace(/((?:api[_-]?key|authorization|token|password|secret)\s*["']?\s*[:=]\s*["']?)[^\s,"'}]+/gi, '$1[REDACTED]')
}

async function errorDetails(response, apiKey) {
  const reader = response.body?.getReader()
  const decoder = new TextDecoder()
  let text = '', length = 0
  try {
    while (reader) {
      const { value, done } = await reader.read()
      if (done) break
      text += decoder.decode(value.subarray(0, 16384 - length), { stream: true })
      length += value.length
      if (length > 16384) break
    }
    text += decoder.decode()
  } catch { /* Retain the HTTP status if the error body cannot be read. */ }
  finally { await reader?.cancel().catch(() => {}) }
  const detail = { http_status: response.status }
  try {
    const payload = JSON.parse(text)
    const error = payload.error ?? payload
    for (const key of ['code', 'type', 'message', 'param']) {
      if (typeof error?.[key] === 'string' || typeof error?.[key] === 'number') {
        detail[key] = redact(error[key], apiKey).slice(0, key === 'message' ? 1200 : 150)
      }
    }
    if (typeof error === 'string') detail.message = redact(error, apiKey).slice(0, 1200)
  } catch { detail.message = 'The endpoint returned a non-JSON error body; inspect the backend HTTP trace.' }
  const requestId = response.headers.get('x-request-id') || response.headers.get('request-id')
  if (requestId) detail.request_id = redact(requestId, apiKey).slice(0, 200)
  const retryHeader = response.headers.get('retry-after')
  const retryAfter = Number(retryHeader)
  const retryMs = Number.isFinite(retryAfter) ? retryAfter * 1000 : Date.parse(retryHeader) - Date.now()
  if (retryMs > 0) detail.retry_after_ms = Math.min(30000, retryMs)
  return detail
}

function httpFailure(detail) {
  const status = detail.http_status
  const description = `${detail.code || ''} ${detail.type || ''} ${detail.message || ''}`
  const authentication = [401, 403].includes(status)
  const quota = /insufficient_quota|quota_exceeded|billing|credit.*(?:exhaust|insufficient)/i.test(description)
  const unsupported = /unsupported|not.supported|unknown.*(?:model|parameter|tool)|model.not.found|no.*(?:channel|provider).*model|web[_ -]?search.*(?:is not enabled|is disabled|not allowed for)|(?:does not|doesn't) support.*web[_ -]?search/i.test(description)
  const queryParameter = /^(?:input|query|queries|messages)(?:$|[.\[])/i.test(String(detail.param || ''))
  const invalidParameter = Boolean(detail.param) || /invalid_parameter|missing_required_parameter|(?:invalid|missing|unrecognized)\s+(?:request\s+)?parameter/i.test(description)
  const protocolUnavailable = [404, 405, 501].includes(status) || (!queryParameter && (unsupported || invalidParameter))
  const rejected = [400, 422].includes(status) && !protocolUnavailable
  const category = authentication ? 'authentication' : quota ? 'quota' : unsupported && !queryParameter ? 'unsupported' :
    protocolUnavailable ? 'protocol' : rejected ? 'request_rejected' :
      status === 429 ? 'rate_limit' : status >= 500 ? 'upstream' : 'request'
  const hint = authentication ? 'Check the search endpoint credentials and permissions.' :
    quota ? 'The search endpoint has no available quota.' : status === 429 ? 'The search endpoint is rate limited.' :
      status >= 500 ? 'The upstream search service failed.' : 'The search request was rejected.'
  return new SearchFailure(`Search returned HTTP ${status}. ${hint}${detail.message ? ' ' + detail.message : ''}`, {
    ...detail, category, scope: quota ? 'configuration' : authentication || protocolUnavailable ? 'protocol' : 'request',
    retryable: !authentication && !quota && !protocolUnavailable && (rejected || [408, 425, 429].includes(status) || status >= 500),
    fatal: authentication || quota,
  })
}

function payloadFailure(payload) {
  // Some compatible gateways return an API error envelope with HTTP 200.
  // Only server error fields count here; model prose never establishes capability.
  if (!payload.error) return null
  const error = payload.error
  const detail = typeof error === 'object' && error !== null ? Object.fromEntries(
    ['code', 'type', 'message', 'param'].filter(key => ['string', 'number'].includes(typeof error[key]))
      .map(key => [key, error[key]])) : { message: String(error).slice(0, 1200) }
  const status = Number(error?.status ?? payload.status_code)
  const category = `${detail.code || ''} ${detail.type || ''}`
  detail.http_status = Number.isInteger(status) && status >= 400 && status < 600 ? status :
    /authentication|invalid_api_key|permission_denied/i.test(category) ? 401 :
    /quota|rate_limit|too_many_requests/i.test(category) ? 429 :
    /server_error|internal_error|overloaded|temporarily_unavailable/i.test(category) ? 503 : 400
  return httpFailure(detail)
}

function cancelled(signal) {
  if (signal?.aborted) throw new SearchFailure('Search was cancelled', { category: 'cancelled', fatal: true })
}

async function pause(milliseconds, signal) {
  cancelled(signal)
  if (!milliseconds) return
  await new Promise((resolve, reject) => {
    const finish = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
    const abort = () => { finish(); reject(new SearchFailure('Search was cancelled', { category: 'cancelled', fatal: true })) }
    const timer = setTimeout(() => { finish(); resolve() }, milliseconds)
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) abort()
  })
}

export function createSearchClient(config, fetchImpl = (...args) => globalThis.fetch(...args)) {
  const protocol = config.protocol || 'auto'
  if (!['auto', ...SEARCH_PROTOCOLS].includes(protocol)) throw new Error('Invalid search API protocol')
  const path = new URL(config.baseURL).pathname
  const preferred = /\/messages\/?$/.test(path) ? 'messages' : /\/chat\/completions\/?$/.test(path) ? 'chat_completions' : 'responses'
  const protocols = protocol === 'auto' ? [preferred, ...SEARCH_PROTOCOLS.filter(item => item !== preferred)] : [protocol]
  const retryDelay = Math.max(0, Math.min(2000, Number(config.retryDelayMs ?? 500)))
  const timeoutMs = Number(config.timeoutMs ?? 120000)
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid search request timeout')
  let selected = null, terminalFailure = null
  const unavailable = new Set()
  const redactFailure = error => new SearchFailure(redact(error.message, config.apiKey), Object.fromEntries(
    Object.entries(error).map(([key, value]) => [key, typeof value === 'string' ? redact(value, config.apiKey) : value])))

  function failedSearch(failures, category) {
    // Only explicit capability/configuration failures are durable. A refused
    // request or a text-only answer cannot disprove a protocol's search support.
    const fatal = failures.length > 0 && failures.every(item => ['protocol', 'configuration'].includes(item.scope))
    const requestFailures = failures.filter(item => !['protocol', 'configuration'].includes(item.scope))
    category ||= fatal ? (failures.every(item => item.category === 'authentication') ? 'authentication' : 'unsupported') :
      requestFailures.length && requestFailures.every(item => item.category === 'rate_limit') ? 'rate_limit' : 'unavailable'
    const reason = fatal ? 'capability_unavailable' : category === 'timeout' ? 'timeout' :
      requestFailures.length && requestFailures.every(item => item.category === 'unverified_retrieval') ? 'unverified_retrieval' : 'search_service_failure'
    const details = { category, reason, retryable: !fatal, fatal, attempts: failures }
    const error = new SearchFailure('Search could not obtain verified retrieval results. ' +
      (reason === 'unverified_retrieval' ? 'The endpoint returned no completed server search evidence; text links cannot verify live retrieval. ' : '') +
      'Inspect backend diagnostics or configure a search endpoint with supported web search.\n' +
      SEARCH_FAILURE_MARKER + JSON.stringify(details), details)
    if (fatal) terminalFailure = error
    return error
  }

  async function execute(query, signal, failures) {
    const order = selected ? [selected, ...protocols.filter(item => item !== selected)] : protocols
    for (const protocol of order.filter(item => !unavailable.has(item))) {
      const request = searchRequest(protocol, config, query)
      let adapted = false
      for (let attempt = 0; attempt < 3; attempt++) {
        cancelled(signal)
        try {
          const response = await fetchImpl(request.url, { method: 'POST', redirect: 'error', signal,
            headers: request.headers, body: JSON.stringify(request.body) })
          if (!response.ok) {
            const detail = await errorDetails(response, config.apiKey)
            // Actual parameter errors, never hostnames/model names, drive adaptation.
            const description = `${detail.param || ''} ${detail.message || ''}`
            if (!adapted && [400, 422].includes(response.status)) {
              if (/thinking|reasoning/i.test(description) && /tool_choice|required|forced/i.test(description)) {
                request.body[protocol === 'messages' ? 'thinking' : 'reasoning'] =
                  protocol === 'messages' ? { type: 'disabled' } : { effort: 'none' }
              } else if (/tool_choice|include|store/.test(description)) {
                delete request.body.tool_choice
                delete request.body.include
                delete request.body.store
              } else { throw httpFailure(detail) }
              adapted = true
              continue
            }
            throw httpFailure(detail)
          }
          let payload
          try { payload = await response.json() } catch { throw new SearchFailure('Search endpoint returned invalid JSON', { category: 'invalid_response', retryable: true }) }
          if (!payload || typeof payload !== 'object') throw new SearchFailure('Search endpoint returned invalid JSON data', { category: 'invalid_response', retryable: true })
          const apiError = payloadFailure(payload)
          if (apiError) throw apiError
          const value = parseSearchResult[protocol](payload)
          selected = protocol
          return { ...value, protocol, ...(failures.length ? { recovered_from: failures } : {}) }
        } catch (rawError) {
          cancelled(signal)
          const error = redactFailure(rawError instanceof SearchFailure ? rawError :
            new SearchFailure('Search endpoint connection failed', { category: 'network', retryable: true }))
          const detail = { protocol, attempt: attempt + 1, category: error.category, scope: error.scope, retryable: error.retryable,
            ...(error.http_status ? { http_status: error.http_status } : {}),
            ...Object.fromEntries(['code', 'type', 'message', 'param', 'request_id'].filter(key => error[key]).map(key => [key, error[key]])) }
          failures.push(detail)
          // In automatic mode, authentication can be specific to an API's
          // header scheme. Try the other documented formats on this SAME origin.
          if (error.fatal && !(protocols.length > 1 && error.category === 'authentication')) {
            terminalFailure = new SearchFailure(error.message + '\n' + SEARCH_FAILURE_MARKER + JSON.stringify({
              category: error.category, reason: error.category, retryable: false, fatal: true, attempts: failures }), { ...error, fatal: true })
            throw terminalFailure
          }
          if (error.retryable && attempt < 2) {
            await pause(error.retry_after_ms ?? Math.min(2000, retryDelay * 2 ** attempt * (.8 + Math.random() * .4)), signal)
            continue
          }
          if (['protocol', 'configuration'].includes(error.scope)) unavailable.add(protocol)
          break
        }
      }
    }
    throw failedSearch(failures)
  }

  return {
    async search(query, signal) {
      cancelled(signal)
      if (terminalFailure) throw terminalFailure
      const controller = new AbortController()
      const abort = () => controller.abort()
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) controller.abort()
      const timer = setTimeout(() => controller.abort(), timeoutMs)
      const failures = []
      try {
        return await execute(query, controller.signal, failures)
      } catch (error) {
        cancelled(signal)
        if (controller.signal.aborted) {
          failures.push({ category: 'timeout', scope: 'request', retryable: true })
          throw failedSearch(failures, 'timeout')
        }
        throw error
      } finally {
        clearTimeout(timer)
        signal?.removeEventListener('abort', abort)
      }
    },
  }
}
