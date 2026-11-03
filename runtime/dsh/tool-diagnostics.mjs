// Model-visible failures, independent of UI/audit telemetry. Never include args,
// stacks, headers or arbitrary error-object fields in the diagnostic envelope.
import { SEARCH_FAILURE_MARKER } from './search-protocols.mjs'

export const DIAGNOSTIC_MARKER = 'TOOL DIAGNOSTIC\n'
export const name = 'resource-to-skill-tool-diagnostics'
export const inject = ['tools']

export function runtimeSecrets() {
  const secrets = [process.env.RESOURCE_TO_SKILL_OPENAI_API_KEY, process.env.DEEPSEEK_API_KEY]
  for (const key of ['RESOURCE_TO_SKILL_OPENAI_SEARCH_CONFIG', 'RESOURCE_TO_SKILL_CAPABILITY_CONFIG']) {
    try { const value = JSON.parse(process.env[key] || '{}'); secrets.push(value.apiKey, value.token) } catch {}
  }
  return secrets.filter(value => typeof value === 'string' && value.length > 5)
}

export function safeText(value, secrets = runtimeSecrets(), limit = 1800) {
  let text = String(value ?? '')
  for (const secret of secrets) text = text.split(secret).join('[REDACTED]')
  return text.replace(/\bBearer\s+[^\s"'<>]+/gi, 'Bearer [REDACTED]')
    .replace(/\bsk-[\w*.-]+/g, '[REDACTED]')
    .replace(/company-free-v1:[^\s"'<>]+/g, '[REDACTED]')
    .replace(/((?:api[_-]?key|authorization|access_token|capability_token|password|secret)["']?\s*[:=]\s*)[^\s,;}]+/gi, '$1[REDACTED]')
    .slice(0, limit)
}

export function diagnosticError(details) {
  const diagnostic = { retryable: false, model_can_fix: false, fatal: true, ...details }
  return new Error(DIAGNOSTIC_MARKER + JSON.stringify(diagnostic))
}

function markerValue(text, marker) {
  try { return JSON.parse(text.split(marker)[1]?.split('\n')[0] || 'null') } catch { return null }
}

export function toolDiagnostic(tool, result, secrets = runtimeSecrets()) {
  if (!result?.isError) return null
  const text = (result.content || []).filter(block => block.type === 'text').map(block => block.text || '').join('\n')
  const raw = markerValue(text, DIAGNOSTIC_MARKER)
  const search = markerValue(text, SEARCH_FAILURE_MARKER)
  const message = safeText(result.error?.message || text || 'Tool execution failed', secrets)
  const code = result.error?.code || ''
  let detail
  if (raw) {
    detail = Object.fromEntries(['origin', 'code', 'field', 'expected', 'problem', 'http_status',
      'retryable', 'model_can_fix', 'fatal', 'action'].filter(key =>
      ['string', 'number', 'boolean'].includes(typeof raw[key])).map(key => [key, raw[key]]))
  } else if (code === 'INVALID_ARGS' || /invalid arguments:|missing required property/i.test(message)) {
    detail = { origin: 'model_arguments', code: 'TOOL_INPUT_SCHEMA_ERROR', problem: message,
      retryable: false, model_can_fix: true, fatal: false,
      action: 'Correct the named missing/invalid fields using the tool schema, then call again. Do not repeat unchanged arguments.' }
  } else if (code === 'INVALID_TOOL_OUTPUT' || /returned invalid output|not lossless JSON/i.test(message)) {
    detail = { origin: 'server_adapter', code: 'TOOL_OUTPUT_SCHEMA_ERROR', problem: message,
      retryable: false, model_can_fix: false, fatal: true,
      action: 'Stop this tool. Its server output violates the contract; changing the query cannot repair it. Preserve existing results and report the failure.' }
  } else if (search) {
    detail = { origin: 'search_backend', code: 'SEARCH_SERVICE_FAILURE', problem: safeText(JSON.stringify(search), secrets),
      retryable: search.retryable === true, model_can_fix: false, fatal: search.fatal === true,
      action: search.fatal ? 'Stop web_search; preserve existing candidates and report the backend error.' :
        'Use only the bounded discovery recovery allowance; inspect existing leads meanwhile.' }
  } else {
    const input = /\[(?:invalid_arguments|invalid_request|validation_error)\]/i.test(message)
    const transient = /timeout|timed out|ECONNRESET|ECONNREFUSED|fetch failed|HTTP (?:408|429|5\d\d)/i.test(message)
    const fatal = /authentication|invalid_api_key|insufficient_quota|budget_exhausted|capability.*(?:expired|denied)|\[(?:unauthorized|forbidden)\]/i.test(message)
    const inaccessible = /HTTP (?:403|404)|status(?:_code)?[":\s]+(?:403|404)/i.test(message)
    detail = { origin: input ? 'model_arguments' : 'tool_execution', code: input ? 'TOOL_INPUT_ERROR' :
      transient ? 'TOOL_TEMPORARY_FAILURE' : inaccessible ? 'SOURCE_UNAVAILABLE' : fatal ? 'TOOL_CONFIGURATION_ERROR' : 'TOOL_EXECUTION_ERROR',
      problem: message, retryable: transient, model_can_fix: input || inaccessible, fatal,
      action: input ? 'Correct the reported fields; do not repeat unchanged arguments.' : inaccessible ?
        'Use an observed public alternative URL; do not repeatedly guess paths.' : fatal ?
        'Stop this tool and report the configuration problem.' :
        'Inspect the error and existing results. Do not repeat identical failing calls; transient recovery is bounded.' }
  }
  return { tool: safeText(tool, secrets, 120), ...Object.fromEntries(Object.entries(detail).map(([key, value]) =>
    [key, typeof value === 'string' ? safeText(value, secrets) : value])) }
}

function callKey(exec) {
  let args = exec.args ?? exec.arguments ?? exec.input ?? {}
  if (typeof args === 'string') { try { args = JSON.parse(args) } catch {} }
  const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ?
    Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value
  return JSON.stringify([exec.name, canonical(args)])
}

export function apply(ctx) {
  const failed = new Map(), blocked = new Map()
  ctx.tools.guard(exec => {
    if (blocked.has(exec.name)) return blocked.get(exec.name)
    const history = failed.get(callKey(exec))
    if (history && history.count >= (history.retryable ? 3 : 2)) return 'Repeated identical failing call blocked. ' +
      history.feedback + '\nChange the invalid arguments or use another approach; preserve useful results and finish.'
  })
  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next()
    if (decision.kind !== 'accept') return decision
    const diagnostic = toolDiagnostic(exec.name, result)
    if (!diagnostic) { failed.delete(callKey(exec)); return decision }
    const feedback = DIAGNOSTIC_MARKER + JSON.stringify(diagnostic)
    const key = callKey(exec), previous = failed.get(key)
    const signature = JSON.stringify([diagnostic.code, diagnostic.field, diagnostic.origin])
    // A changed failure is new evidence; the same failure must not loop forever.
    failed.set(key, { signature, count: previous?.signature === signature ? previous.count + 1 : 1,
      retryable: diagnostic.retryable, feedback })
    if (failed.size > 200) failed.delete(failed.keys().next().value)
    if (diagnostic.fatal) blocked.set(exec.name, feedback)
    const content = [{ type: 'text', text: feedback }]
    if (exec.name === 'web_search') content.push({ type: 'text', text: SEARCH_FAILURE_MARKER + JSON.stringify({
      category: diagnostic.origin === 'model_arguments' ? 'model_arguments' : 'search_service_failure',
      retryable: diagnostic.retryable, fatal: diagnostic.fatal, model_can_fix: diagnostic.model_can_fix }) })
    // Preserve host budget feedback while removing unfiltered upstream error text.
    for (const block of decision.content || []) if (block.type === 'text' && block.text?.startsWith('DISCOVERY BUDGET UPDATE')) content.push(block)
    return { ...decision, content }
  })
}
