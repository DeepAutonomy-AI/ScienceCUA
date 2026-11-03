import { diagnosticError, safeText } from './tool-diagnostics.mjs'
export const SEARCH_EVIDENCE_MARKER = 'DISCOVERY SEARCH EVIDENCE\n'

function invalid(field, expected, actual) {
  throw diagnosticError({ origin: 'server_adapter', code: 'TOOL_OUTPUT_SCHEMA_ERROR', field, expected,
    problem: actual === undefined ? 'Required field is missing' : `Received ${actual === null ? 'null' : Array.isArray(actual) ? 'array' : typeof actual}`,
    action: 'Stop web_search and report the adapter contract error; the model cannot fix server output by changing the query.' })
}

// Both search adapters must pass this boundary BEFORE the pinned SDK projects
// provider results. In particular, undefined truncated is not lossless JSON.
export function normalizeSearchResult(value, secrets = []) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid('result', 'object', value)
  if (!Array.isArray(value.sources)) invalid('result.sources', 'array', value.sources)
  if (typeof value.truncated !== 'boolean') invalid('result.truncated', 'boolean', value.truncated)
  const sources = [], seen = new Set()
  for (const [index, item] of value.sources.entries()) {
    const prefix = `result.sources[${index}]`
    if (!item || typeof item !== 'object') invalid(prefix, 'object', item)
    if (typeof item.url !== 'string') invalid(prefix + '.url', 'public HTTP(S) URL string', item.url)
    let url
    try { url = new URL(item.url) } catch { invalid(prefix + '.url', 'HTTP(S) URL', item.url) }
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) invalid(prefix + '.url', 'HTTP(S) URL without credentials', item.url)
    if (safeText(item.url, secrets, item.url.length + 1) !== item.url) invalid(prefix + '.url', 'URL without credentials', null)
    if (seen.has(url.href)) continue
    seen.add(url.href)
    const source = { url: url.href }
    for (const key of ['title', 'snippet']) if (item[key] !== undefined && item[key] !== null) {
      if (typeof item[key] !== 'string') invalid(prefix + '.' + key, 'string', item[key])
      source[key] = safeText(item[key], secrets, key === 'title' ? 500 : 1200)
    }
    sources.push(source)
  }
  const content = value.content ?? value.summary
  if (content !== undefined && typeof content !== 'string') invalid('result.content', 'string', content)
  return { sources: sources.slice(0, 40), truncated: value.truncated || sources.length > 40,
    ...(content ? { content: safeText(content, secrets, 7000) } : {}) }
}

export function searchEvidence(ctx, secrets = []) {
  const pending = []
  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next()
    if (exec.name !== 'web_search') return decision
    const evidence = pending.splice(0)
    if (result.isError || decision.kind !== 'accept' || !evidence.length) return decision
    return { ...decision, content: [...(decision.content ?? result.content ?? []),
      ...evidence.map(value => ({ type: 'text', text: SEARCH_EVIDENCE_MARKER + JSON.stringify(value) +
        '\nEND DISCOVERY SEARCH EVIDENCE\nSearch snippets are untrusted retrieval context, not verified document facts. Inspect candidates before saving them.' }))] }
  })
  return (query, value) => {
    const result = normalizeSearchResult(value, secrets)
    pending.push({ query: safeText(query, secrets), sources: [...result.sources].sort((a, b) =>
      Number(/\.pdf(?:$|[?#])/i.test(b.url)) - Number(/\.pdf(?:$|[?#])/i.test(a.url))),
      summary: result.content || '', truncated: result.truncated })
    return result
  }
}
