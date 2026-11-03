// External plugins live outside the generated DSH profile, so a bare package
// import cannot be resolved from this file's directory. The pinned worker passes
// the exact profile-owned entry point without copying or modifying DSH code.
import { pathToFileURL } from 'node:url'
import { installDiscoveryBudget } from './discovery-budget-policy.mjs'

const toolsModule = process.env.RESOURCE_TO_SKILL_DSH_TOOLS_MODULE
if (!toolsModule) throw new Error('RESOURCE_TO_SKILL_DSH_TOOLS_MODULE is required')
const { defineTool } = await import(pathToFileURL(toolsModule).href)

export const name = 'resource-to-skill-capability-proxy'
export const inject = ['tools']

const CONFIG_ENV = 'RESOURCE_TO_SKILL_CAPABILITY_CONFIG'

function parseConfig() {
  const raw = process.env[CONFIG_ENV]
  if (!raw) return null
  let value
  try {
    value = JSON.parse(raw)
  } catch {
    throw new Error(`${CONFIG_ENV} must be valid JSON`)
  }
  if (!value || typeof value !== 'object') throw new Error('Capability config must be an object')
  if (typeof value.gatewayUrl !== 'string' || !value.gatewayUrl.startsWith('http')) {
    throw new Error('Capability gateway URL is missing')
  }
  if (typeof value.token !== 'string' || value.token.length < 32) {
    throw new Error('Capability token is missing')
  }
  if (!Array.isArray(value.toolSpecs) || !value.context || typeof value.context !== 'object') {
    throw new Error('Capability tool specs or context are missing')
  }
  return value
}

function convertSchema(schema, required = false) {
  const source = schema && typeof schema === 'object' ? schema : {}
  const converted = {
    type: source.type || 'string',
  }
  // dsh-tools treats `required` as a presence flag: when emitted it must be
  // literal true.  Optional parameters therefore omit the field entirely.
  if (required) converted.required = true
  if (typeof source.description === 'string') converted.description = source.description
  if (Array.isArray(source.enum)) converted.enum = source.enum
  if (source.default !== undefined) converted.default = source.default
  // The pinned dsh-tools value-schema DSL does not accept JSON Schema numeric
  // or string-length bounds. These bounds remain enforced by the Capability
  // Gateway/Python handler, so omit them at this adapter boundary instead of
  // preventing the plugin from loading.
  if (source.items) converted.items = convertSchema(source.items, false)
  if (source.properties && typeof source.properties === 'object') {
    converted.properties = convertParameters(source)
    converted.additionalProperties = source.additionalProperties === true
  } else if (converted.type === 'object') {
    // The pinned dsh-tools schema compiler requires an explicit object policy.
    converted.additionalProperties = source.additionalProperties === true
  }
  return converted
}

function convertParameters(schema) {
  const properties = schema && typeof schema.properties === 'object' ? schema.properties : {}
  const required = new Set(Array.isArray(schema?.required) ? schema.required : [])
  return Object.fromEntries(
    Object.entries(properties).map(([key, value]) => [key, convertSchema(value, required.has(key))]),
  )
}

function normalizeSpec(spec) {
  const fn = spec?.type === 'function' ? spec.function : spec
  if (!fn || typeof fn.name !== 'string' || !fn.name) throw new Error('Invalid tool specification')
  return {
    name: fn.name,
    description: typeof fn.description === 'string' ? fn.description : '',
    parameters: convertParameters(fn.parameters || { type: 'object', properties: {} }),
  }
}

function normalizeValue(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value
  return { ok: true, value }
}

function callId(exec) {
  const supplied = exec?.toolCallId || exec?.callId || exec?.id
  if (typeof supplied === 'string' && /^[A-Za-z0-9._:-]+$/.test(supplied)) return supplied
  return `tool-${crypto.randomUUID()}`
}

export function apply(ctx) {
  const config = parseConfig()
  if (!config) return
  if (config.discoveryBudget) installDiscoveryBudget(ctx, config.discoveryBudget)
  const seen = new Set()
  for (const rawSpec of config.toolSpecs) {
    const spec = normalizeSpec(rawSpec)
    if (seen.has(spec.name)) throw new Error(`Duplicate capability tool: ${spec.name}`)
    seen.add(spec.name)
    ctx.tools.register(
      defineTool({
        name: spec.name,
        description: spec.description,
        parameters: spec.parameters,
        output: {
          schema: { type: 'object', additionalProperties: true },
          render: (_args, value) => [
            { type: 'text', text: JSON.stringify(normalizeValue(value)) },
          ],
        },
        async execute(args, exec) {
          const response = await fetch(config.gatewayUrl, {
            method: 'POST',
            headers: {
              authorization: `Bearer ${config.token}`,
              'content-type': 'application/json',
            },
            body: JSON.stringify({
              ...config.context,
              tool_name: spec.name,
              arguments: args || {},
              call_id: callId(exec),
            }),
          })
          let payload
          try {
            payload = await response.json()
          } catch {
            throw new Error(`Capability gateway returned HTTP ${response.status}`)
          }
          if (!response.ok || !payload?.ok) {
            const code = payload?.error?.code || `http_${response.status}`
            const message = payload?.error?.message || 'Capability gateway denied the tool call'
            throw new Error(`[${code}] ${message}`)
          }
          return normalizeValue(payload.result)
        },
      }),
    )
  }
}
