// Capture model HTTP bodies before the SDK parses them. This plugin never sends
// diagnostics to the model, browser, or exported knowledge artifacts.
import { createHash, randomUUID } from 'node:crypto'
import { appendFileSync, mkdirSync, renameSync, writeFileSync } from 'node:fs'
import { join, toNamespacedPath } from 'node:path'

export const name = 'resource-to-skill-http-trace'

function patternsFor(secrets) {
  return [...new Set(secrets.filter(value => typeof value === 'string' && value).flatMap(value => [
    value, JSON.stringify(value).slice(1, -1), encodeURIComponent(value),
  ]))].sort((a, b) => b.length - a.length)
}

function scrubber(patterns) {
  const hold = Math.max(0, ...patterns.map(value => value.length - 1))
  let pending = ''
  const clean = text => {
    for (const pattern of patterns) text = text.split(pattern).join('[REDACTED]')
    return text
  }
  return {
    clean,
    push(text, final = false) {
      pending = clean(pending + text)
      const end = final ? pending.length : Math.max(0, pending.length - hold)
      const value = pending.slice(0, end)
      pending = pending.slice(end)
      return value
    },
  }
}

function headersForTrace(headers) {
  return Object.fromEntries(['content-type', 'content-length', 'content-encoding',
    'x-request-id', 'request-id', 'traceparent'].flatMap(name => {
    const value = headers.get(name)
    return value === null ? [] : [[name, value]]
  }))
}

async function requestProjection(input, init) {
  let body = init?.body
  if (body === undefined && input instanceof Request) {
    // A clone preserves the actual outgoing body. Bound inspection separately
    // from response capture; large prompts/images need not be copied to logs.
    const reader = input.clone().body?.getReader()
    if (reader) {
      let length = 0
      const chunks = []
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          length += value.byteLength
          if (length > 16 * 1024 * 1024) return { body_projection: 'request_too_large' }
          chunks.push(Buffer.from(value))
        }
        body = Buffer.concat(chunks).toString('utf8')
      } finally {
        // Do not await tee cancellation: the other branch has not been sent yet.
        void reader.cancel().catch(() => {})
      }
    }
  }
  if (typeof body !== 'string' || body.length > 16 * 1024 * 1024) {
    return { body_projection: 'unavailable' }
  }
  try {
    const value = JSON.parse(body)
    return {
      body_projection: 'tool_definitions_only',
      model: value.model, stream: value.stream,
      tools: value.tools, tool_choice: value.tool_choice,
      request_body_sha256: createHash('sha256').update(body).digest('hex'),
    }
  } catch {
    return { body_projection: 'not_json' }
  }
}

export function createTraceFetch(config, fetchImpl = globalThis.fetch) {
  const responseLimit = Number(config.max_response_bytes)
  const sessionLimit = Number(config.max_session_bytes)
  if (!config.directory || !Number.isSafeInteger(responseLimit) || responseLimit < 1 ||
      !Number.isSafeInteger(sessionLimit) || sessionLimit < 1) {
    throw new Error('Invalid HTTP trace configuration')
  }
  const directory = toNamespacedPath(config.directory)
  let sequence = 0, sessionBytes = 0

  return async function tracedFetch(input, init) {
    let url
    try { url = new URL(input instanceof Request ? input.url : String(input)) } catch {
      return fetchImpl(input, init)
    }
    const method = String(init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase()
    if (method !== 'POST' || !['http:', 'https:'].includes(url.protocol) ||
        !/\/(?:chat\/completions|responses|messages)\/?$/.test(url.pathname)) {
      return fetchImpl(input, init)
    }
    const secrets = [...(config.secrets || [])]
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined))
    for (const key of ['authorization', 'x-api-key', 'api-key']) {
      const value = headers.get(key)
      if (value) secrets.push(value, value.replace(/^Bearer\s+/i, ''))
    }
    const redact = scrubber(patternsFor(secrets))
    let record, bodyPath, metaPath
    const saveMeta = () => {
      const temporary = `${metaPath}.tmp`
      writeFileSync(temporary, redact.clean(JSON.stringify(record, null, 2)), { mode: 0o600 })
      renameSync(temporary, metaPath)
    }
    try {
      mkdirSync(directory, { recursive: true })
      const traceId = `http_${String(++sequence).padStart(6, '0')}_${randomUUID().slice(0, 8)}`
      const stem = join(directory, traceId)
      bodyPath = `${stem}.response.body`
      metaPath = `${stem}.meta.json`
      record = {
        schema_version: 'model-http-trace.v1', trace_id: traceId,
        run_id: config.run_id, child_session_id: config.session_id, stage: config.stage,
        started_at: new Date().toISOString(),
        // URL credentials, query strings, and request headers are never retained.
        url: `${url.origin}${url.pathname}`, method,
        response_body_file: `${traceId}.response.body`, state: 'requesting',
        complete: false, truncated: false, received_bytes: 0, captured_bytes: 0,
        response_limit_bytes: responseLimit, session_limit_bytes: sessionLimit,
        redacted: true,
      }
      writeFileSync(`${stem}.request.json`, redact.clean(JSON.stringify({
        ...record, ...await requestProjection(input, init),
      }, null, 2)), { mode: 0o600 })
      writeFileSync(bodyPath, '', { mode: 0o600 })
      saveMeta()
    } catch {
      // A diagnostic failure must never replace a successful provider response.
      record = null
    }
    let response
    try {
      response = await fetchImpl(input, init)
    } catch (error) {
      if (record) {
        record.state = 'network_error'
        record.error = String(error)
        record.finished_at = new Date().toISOString()
        try { saveMeta() } catch {}
      }
      throw error
    }
    if (!record) return response
    record.status = response.status
    record.response_headers = headersForTrace(response.headers)
    record.state = 'receiving'
    try { saveMeta() } catch {}
    const decoder = new TextDecoder()
    const hash = createHash('sha256')
    let finished = false, captureFailed = false
    const capture = text => {
      if (!text || captureFailed) return
      const bytes = Buffer.from(text)
      const available = Math.max(0, Math.min(
        responseLimit - record.captured_bytes, sessionLimit - sessionBytes,
      ))
      const length = Math.min(available, bytes.length)
      if (length < bytes.length) record.truncated = true
      if (length) {
        try {
          appendFileSync(bodyPath, bytes.subarray(0, length))
          record.captured_bytes += length
          sessionBytes += length
        } catch {
          captureFailed = true
          record.capture_error = true
        }
      }
    }
    const finish = (state, error) => {
      if (finished) return
      finished = true
      capture(redact.push(decoder.decode(), true))
      record.state = state
      record.complete = state === 'complete'
      record.response_body_sha256 = hash.digest('hex')
      record.finished_at = new Date().toISOString()
      if (error) record.error = String(error)
      try { saveMeta() } catch {}
    }
    if (!response.body) {
      finish('complete')
      return response
    }
    const reader = response.body.getReader()
    const body = new ReadableStream({
      async pull(controller) {
        let item
        try { item = await reader.read() } catch (error) {
          finish('read_error', error)
          controller.error(error)
          return
        }
        if (item.done) {
          finish('complete')
          controller.close()
          return
        }
        // Persist before passing identical bytes to the SDK. Preserve stream
        // order and backpressure; never accumulate the full response in RAM.
        record.received_bytes += item.value.byteLength
        hash.update(item.value)
        capture(redact.push(decoder.decode(item.value, { stream: true })))
        controller.enqueue(item.value)
      },
      async cancel(reason) {
        finish('cancelled', reason)
        await reader.cancel(reason)
      },
    }, { highWaterMark: 0 })
    const wrapped = new Response(body, {
      status: response.status, statusText: response.statusText, headers: response.headers,
    })
    Object.defineProperties(wrapped, {
      url: { value: response.url }, redirected: { value: response.redirected }, type: { value: response.type },
    })
    return wrapped
  }
}

export function apply(ctx) {
  const raw = process.env.RESOURCE_TO_SKILL_HTTP_TRACE_CONFIG
  if (!raw) return
  const original = globalThis.fetch
  globalThis.fetch = createTraceFetch(JSON.parse(raw), original)
  ctx.on('dispose', () => { globalThis.fetch = original })
}
