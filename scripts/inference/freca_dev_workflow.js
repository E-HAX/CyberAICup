export const meta = {
  name: 'freca-dev-inference',
  description: 'Standard pass (all 41 CPs) + confidence-gated escalation to adversarial/stepbystep + tiebreak',
  phases: [
    { title: 'Standard pass' },
    { title: 'Escalation' },
    { title: 'Tiebreak' },
  ],
}

const ROOT = args.root
const CASE_IDS = args.caseIds
const T = args.templates
const MODEL = args.model || 'haiku'
const CONF_THRESHOLD = args.confidenceThreshold ?? 0.85
const ALL_CPS = args.cps || Array.from({ length: 41 }, (_, i) => `CP${i + 1}`)
const VALID = ['1', '0', 'N/A']

function evidenceFile(caseId) { return `${ROOT}/dev/evidence/${caseId}.md` }
const REFERENCE_FILE = `${ROOT}/okf/reference.md`

function contextBlock(caseId) {
  return `## Context\n\n1. Read the full case evidence file: \`${evidenceFile(caseId)}\`\n` +
    `2. Read the OKF reference bundle (official text for every CP + which policy sections govern it, followed by the full text of those sections): \`${REFERENCE_FILE}\``
}

function render(tpl, vars) {
  let out = tpl
  for (const [k, v] of Object.entries(vars)) out = out.split(`{${k}}`).join(v)
  return out
}

function scopeText(cps) {
  if (cps.length >= 41) return 'all 41 checking points (CP1 through CP41)'
  return `these ${cps.length} checking point(s): ${cps.join(', ')}`
}

const PERSONA_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          cp: { type: 'string' },
          reasoning: { type: 'string', maxLength: 400 },
          verdict: { type: 'string', enum: ['1', '0', 'N/A'] },
          confidence: { type: 'number' },
          cited_evidence: { type: 'string', maxLength: 300 },
        },
        required: ['cp', 'reasoning', 'verdict', 'confidence'],
      },
    },
  },
  required: ['results'],
}

const TIEBREAK_SCHEMA = {
  type: 'object',
  properties: {
    cp: { type: 'string' },
    reasoning: { type: 'string', maxLength: 400 },
    verdict: { type: 'string', enum: ['1', '0', 'N/A'] },
    confidence: { type: 'number' },
    cited_evidence: { type: 'string', maxLength: 300 },
  },
  required: ['cp', 'reasoning', 'verdict', 'confidence'],
}

async function runPersona(caseId, persona, cps) {
  const prompt = render(T[persona], {
    CONTEXT_BLOCK: contextBlock(caseId),
    CP_SCOPE_TEXT: scopeText(cps),
  })
  const res = await agent(prompt, {
    label: `${persona}:${caseId}:${cps.length}cp`,
    phase: cps.length >= 41 ? 'Standard pass' : 'Escalation',
    schema: PERSONA_SCHEMA,
    model: MODEL,
  })
  const map = {}
  for (const r of (res?.results || [])) {
    if (VALID.includes(r.verdict)) map[r.cp] = r
  }
  return map
}

async function runTiebreak(caseId, cp) {
  const prompt = render(T.tiebreaker, { CONTEXT_BLOCK: contextBlock(caseId), CP: cp })
  return agent(prompt, {
    label: `tiebreak:${caseId}:${cp}`,
    phase: 'Tiebreak',
    schema: TIEBREAK_SCHEMA,
    model: MODEL,
  })
}

const results = await pipeline(
  CASE_IDS,
  // stage 1: standard pass, all 41 CPs
  async (_, caseId) => {
    const standard = await runPersona(caseId, 'standard', ALL_CPS)
    return { standard }
  },
  // stage 2: escalate low-confidence CPs to adversarial + stepbystep
  async (prev, caseId) => {
    const { standard } = prev
    const escalate = ALL_CPS.filter(cp => !standard[cp] || standard[cp].confidence < CONF_THRESHOLD)
    if (!escalate.length) return { ...prev, adversarial: {}, stepbystep: {}, escalate }
    const [adversarial, stepbystep] = await parallel([
      () => runPersona(caseId, 'adversarial', escalate),
      () => runPersona(caseId, 'stepbystep', escalate),
    ])
    return { ...prev, adversarial: adversarial || {}, stepbystep: stepbystep || {}, escalate }
  },
  // stage 3: arbitrate
  async (prev, caseId) => {
    const { standard, adversarial, stepbystep, escalate } = prev
    const escalateSet = new Set(escalate)
    const verdicts = {}
    const splits = []

    for (const cp of ALL_CPS) {
      if (!escalateSet.has(cp)) {
        const s = standard[cp]
        verdicts[cp] = s
          ? { verdict: s.verdict, confidence: s.confidence, source: 'single_high_confidence' }
          : { verdict: '0', confidence: 0.2, source: 'missing_standard_vote' }
        continue
      }
      const raw = [standard[cp]?.verdict, adversarial[cp]?.verdict, stepbystep[cp]?.verdict].filter(Boolean)
      if (raw.length < 3) {
        verdicts[cp] = { verdict: raw[0] || '0', confidence: 0.3, source: 'incomplete_escalation_votes', votes: raw }
        continue
      }
      const counts = { '1': 0, '0': 0, 'N/A': 0 }
      for (const v of raw) counts[v]++
      const majorityClass = Object.entries(counts).find(([, c]) => c >= 2)
      if (majorityClass) {
        verdicts[cp] = {
          verdict: majorityClass[0],
          confidence: majorityClass[1] === 3 ? 1.0 : 0.7,
          source: majorityClass[1] === 3 ? 'unanimous' : 'majority',
          votes: raw,
        }
      } else {
        splits.push(cp)
        verdicts[cp] = { verdict: raw[0], confidence: 0.4, source: 'split_pending_tiebreak', votes: raw }
      }
    }

    if (splits.length) {
      const tb = await parallel(splits.map(cp => () => runTiebreak(caseId, cp).then(r => ({ cp, r }))))
      for (const t of tb) {
        if (!t || !t.r) continue
        verdicts[t.cp] = { verdict: t.r.verdict, confidence: t.r.confidence, source: 'tiebreak', reasoning: t.r.reasoning }
      }
    }
    return { caseId, escalatedCount: escalate.length, verdicts }
  },
)

return results
