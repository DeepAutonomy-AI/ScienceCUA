// Search protocol negotiation behind DSH's unchanged native web_search tool.
import { createSearchClient } from './search-request.mjs'
import { searchEvidence } from './search-result-contract.mjs'
export { SEARCH_EVIDENCE_MARKER } from './search-result-contract.mjs'
export { searchSources } from './search-protocols.mjs'
export const name = 'resource-to-skill-openai-search'
export const inject = ['web']

export function apply(ctx) {
  const raw = process.env.RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG
  if (!raw) throw new Error('Search configuration is missing')
  const config = JSON.parse(raw)
  const client = createSearchClient(config)
  const record = searchEvidence(ctx, [config.apiKey])
  ctx.web.registerSearchProvider({
    // Keep the registered ID stable for existing DSH profiles.
    id: 'openai-responses',
    available: () => Boolean(config.apiKey && config.model && config.baseURL),
    async search(request, signal) {
      const value = await client.search(request.query, signal)
      return record(request.query, value)
    },
  })
}
