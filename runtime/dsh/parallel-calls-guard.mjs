// Force parallel_tool_calls: false on tool-bearing chat requests. The relay
// drops tool-call arguments on the streaming parallel path, so the model must
// emit one tool call at a time. Inserted only when the worker opts in.
export const name = 'resource-to-skill-parallel-calls-guard'

const TARGET_PATH = /\/(?:chat\/completions|responses|messages)\/?$/

async function requestBody(input, init) {
  if (typeof init?.body === 'string') return init.body
  if (input instanceof Request) return input.clone().text()
  return null
}

export function apply(ctx) {
  const original = globalThis.fetch
  globalThis.fetch = async function guardedFetch(input, init) {
    let url
    try { url = new URL(input instanceof Request ? input.url : String(input)) } catch {
      return original(input, init)
    }
    const method = String(init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase()
    if (method !== 'POST' || !['http:', 'https:'].includes(url.protocol) || !TARGET_PATH.test(url.pathname)) {
      return original(input, init)
    }
    try {
      const raw = await requestBody(input, init)
      if (typeof raw !== 'string') return original(input, init)
      const payload = JSON.parse(raw)
      if (!Array.isArray(payload.tools) || payload.parallel_tool_calls === false) {
        return original(input, init)
      }
      payload.parallel_tool_calls = false
      const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined))
      headers.delete('content-length')
      const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined)
      return original(new Request(url, {
        method: 'POST', headers, body: JSON.stringify(payload), signal,
      }))
    } catch {
      // A guard failure must never block the provider request.
      return original(input, init)
    }
  }
  ctx.on('dispose', () => { globalThis.fetch = original })
}
