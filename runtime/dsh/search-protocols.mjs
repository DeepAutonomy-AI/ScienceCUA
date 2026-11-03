// Normalize only server-provided retrieval evidence. Plain model links are not sources.
export const SEARCH_PROTOCOLS = ['responses', 'messages', 'chat_completions']
export const SEARCH_FAILURE_MARKER = 'DISCOVERY SEARCH FAILURE\n'

export class SearchFailure extends Error {
  constructor(message, details = {}) {
    super(message)
    const { message: _ignored, ...fields } = details
    Object.assign(this, { category: 'unverified_retrieval', retryable: false, scope: 'request', ...fields })
  }
}

function publicUrl(value) {
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null
    url.hash = ''
    for (const key of [...url.searchParams.keys()]) {
      if (/^utm_/i.test(key) || ['gclid', 'fbclid'].includes(key)) url.searchParams.delete(key)
    }
    return url.href
  } catch { return null }
}

function normalized(sources, summaries) {
  const unique = new Map()
  for (const item of sources) {
    const url = publicUrl(item?.url)
    if (!url) continue
    const source = unique.get(url) || { url }
    const title = typeof item.title === 'string' ? item.title.trim().slice(0, 500) : ''
    const host = new URL(url).hostname.replace(/^www\./, '')
    const quality = text => !text ? 0 : [url, host, `www.${host}`].includes(text) ? 1 : 2
    if (quality(title) > quality(source.title)) source.title = title
    const snippet = item.snippet || item.description
    if (typeof snippet === 'string' && snippet.trim()) source.snippet = snippet.trim().slice(0, 1200)
    unique.set(url, source)
  }
  const summary = summaries.join('\n').replace(/https?:\/\/[^\s<>"\])]+/g,
    value => unique.has(publicUrl(value)) ? value : '[unverified URL omitted]').slice(0, 7000)
  return { sources: [...unique.values()], truncated: false, ...(summary ? { summary } : {}) }
}

export function searchSources(payload) {
  if (payload.status && payload.status !== 'completed') throw new SearchFailure('Responses search did not complete')
  const output = Array.isArray(payload.output) ? payload.output : []
  if (!output.some(item => item.type === 'web_search_call' && item.status === 'completed')) {
    throw new SearchFailure('The endpoint did not execute web_search; generated text is not a search result')
  }
  const sources = [], summaries = []
  for (const item of output) {
    if (item.type === 'web_search_call') sources.push(...(item.action?.sources || []))
    if (item.type === 'message') for (const block of item.content || []) {
      sources.push(...(block.annotations || []).filter(item => item.type === 'url_citation'))
      if (typeof block.text === 'string') summaries.push(block.text)
    }
  }
  return normalized(sources, summaries)
}

export function messagesSearchSources(payload) {
  const blocks = Array.isArray(payload.content) ? payload.content : []
  const results = blocks.filter(item => item.type === 'web_search_tool_result')
  if (!results.length) throw new SearchFailure('The Messages endpoint did not execute web_search')
  const sources = []
  for (const result of results) {
    if (result.content?.type === 'web_search_tool_result_error') {
      const code = result.content.error_code || 'search_error'
      const unsupported = ['web_search_disabled', 'unsupported_tool', 'tool_not_supported'].includes(code)
      throw new SearchFailure(`Search tool returned ${String(code).slice(0, 100)}`,
        { category: unsupported ? 'unsupported' : code === 'too_many_requests' ? 'rate_limit' : 'search_tool', scope: unsupported ? 'protocol' : 'request',
          retryable: ['too_many_requests', 'unavailable', 'internal_error'].includes(code), code })
    }
    if (!Array.isArray(result.content)) throw new SearchFailure('Messages search returned invalid retrieval results')
    sources.push(...result.content.filter(item => item.type === 'web_search_result'))
  }
  return normalized(sources, blocks.filter(item => item.type === 'text' && typeof item.text === 'string').map(item => item.text))
}

export function chatSearchSources(payload) {
  const messages = (payload.choices || []).map(item => item.message || {})
  const sources = messages.flatMap(item => item.annotations || [])
    .filter(item => item.type === 'url_citation').map(item => item.url_citation || item)
  if (!sources.length) throw new SearchFailure('The Chat endpoint returned no structured search citations; generated links are not verified retrieval')
  return normalized(sources, messages.filter(item => typeof item.content === 'string').map(item => item.content))
}

export function searchEndpoint(baseURL, protocol) {
  const url = new URL(baseURL)
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new SearchFailure('Search Base URL must be an HTTP(S) address without credentials, query or fragment', { category: 'configuration' })
  }
  url.pathname = url.pathname.replace(/\/(?:responses|messages|chat\/completions)\/?$/, '').replace(/\/$/, '') +
    '/' + ({ responses: 'responses', messages: 'messages', chat_completions: 'chat/completions' }[protocol])
  return url.href
}

const instructions = 'Find software manuals and tutorials matching the query. Execute web search and return descriptive document titles, actual versions when stated, short relevant excerpts, and cited source URLs. Distinguish unrelated software with the same name. Do not invent compatibility or URLs.'

export function searchRequest(protocol, config, query) {
  const shared = { model: config.model, stream: false }
  const input = `Search the web for: ${query}`
  const headers = { 'Content-Type': 'application/json' }
  let body
  if (protocol === 'messages') {
    headers['x-api-key'] = config.apiKey
    headers['anthropic-version'] = '2023-06-01'
    body = { ...shared, max_tokens: 2048, system: instructions,
      messages: [{ role: 'user', content: input }],
      tools: [{ type: 'web_search_20250305', name: 'web_search', max_uses: 1 }],
      tool_choice: { type: 'tool', name: 'web_search' } }
  } else {
    headers.Authorization = `Bearer ${config.apiKey}`
    body = protocol === 'responses' ? { ...shared, instructions, input, max_output_tokens: 2048,
      tools: [{ type: 'web_search' }], tool_choice: 'required',
      include: ['web_search_call.action.sources'], store: false } : {
      ...shared, messages: [{ role: 'system', content: instructions }, { role: 'user', content: input }],
      web_search_options: {} }
  }
  return { url: searchEndpoint(config.baseURL, protocol), headers, body }
}

export const parseSearchResult = { responses: searchSources, messages: messagesSearchSources, chat_completions: chatSearchSources }
