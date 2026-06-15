// Props:
//   metrics:            EncounterMetrics object | null
//   pipeline:           "whisper" | "lfm" | "hybrid"
//   onHighlightSegment: (segmentIndex: number) => void

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts'

// ---------------------------------------------------------------------------
// Shared shell components
// ---------------------------------------------------------------------------

function PanelShell({ title, children }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 flex flex-col overflow-hidden">
      <div className="px-4 py-2.5 border-b border-gray-100 shrink-0">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          {title}
        </h3>
      </div>
      <div className="flex-1 overflow-auto p-3">{children}</div>
    </div>
  )
}

function PanelSkeleton({ title }) {
  return (
    <PanelShell title={title}>
      <div className="h-full min-h-[110px] bg-gray-100 rounded-lg animate-pulse" />
    </PanelShell>
  )
}

function Unavailable() {
  return (
    <p className="text-sm text-gray-400 text-center mt-6">
      Metric unavailable for this encounter
    </p>
  )
}

// ---------------------------------------------------------------------------
// Panel 1 — Latency Waterfall
// ---------------------------------------------------------------------------

const LATENCY_STAGES = [
  { key: 'asr',  label: 'ASR',         color: '#3b82f6' },
  { key: 'diar', label: 'Diarization', color: '#8b5cf6' },
  { key: 'soap', label: 'SOAP Gen',    color: '#22c55e' },
  { key: 'eval', label: 'Eval',        color: '#f97316' },
]

function LatencyPanel({ metrics }) {
  if (!metrics) return <PanelSkeleton title="Pipeline Latency" />

  const { asr_ms, diarization_ms, note_gen_ms, total_ms } = metrics

  if (asr_ms == null) {
    return <PanelShell title="Pipeline Latency"><Unavailable /></PanelShell>
  }

  const evalMs = Math.max(
    0,
    (total_ms ?? 0) - (asr_ms ?? 0) - (diarization_ms ?? 0) - (note_gen_ms ?? 0),
  )

  const data = [{
    name: 'Pipeline',
    asr:  Math.round(asr_ms ?? 0),
    diar: Math.round(diarization_ms ?? 0),
    soap: Math.round(note_gen_ms ?? 0),
    eval: Math.round(evalMs),
  }]

  return (
    <PanelShell title="Pipeline Latency">
      <div className="flex justify-end mb-1">
        <span className="text-xs text-gray-500 tabular-nums">
          Total: <span className="font-semibold text-gray-800">{total_ms?.toFixed(0)} ms</span>
        </span>
      </div>

      <ResponsiveContainer width="100%" height={72}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
        >
          <XAxis
            type="number"
            tick={{ fontSize: 10, fill: '#9ca3af' }}
            tickFormatter={(v) => `${v}ms`}
            axisLine={false}
            tickLine={false}
          />
          <YAxis type="category" dataKey="name" hide />
          <Tooltip
            formatter={(value, name) => [`${value} ms`, name]}
            contentStyle={{ fontSize: 12 }}
          />
          {LATENCY_STAGES.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} stackId="a" fill={s.color} radius={0} />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-3 mt-2">
        {LATENCY_STAGES.map((s) => (
          <span key={s.key} className="flex items-center gap-1 text-xs text-gray-500">
            <span className="w-2 h-2 rounded-sm shrink-0" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </PanelShell>
  )
}

// ---------------------------------------------------------------------------
// Panel 2 — Medical Concept WER
// ---------------------------------------------------------------------------

const WER_ROWS = [
  { label: 'Medications', key: 'medical_wer_medications' },
  { label: 'Symptoms',    key: 'medical_wer_symptoms'    },
  { label: 'Procedures',  key: 'medical_wer_procedures'  },
]

function WerDot({ value }) {
  if (value == null) {
    return (
      <span className="text-xs text-gray-400 italic">N/A (no ground truth)</span>
    )
  }
  const dotCls =
    value < 0.10 ? 'bg-green-500' :
    value < 0.20 ? 'bg-yellow-400' :
                   'bg-red-500'
  return (
    <span className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full shrink-0 ${dotCls}`} />
      <span className="text-sm tabular-nums text-gray-800">
        {(value * 100).toFixed(1)}%
      </span>
    </span>
  )
}

function WerPanel({ metrics }) {
  if (!metrics) return <PanelSkeleton title="Medical ASR Accuracy" />

  const allNull = WER_ROWS.every((r) => metrics[r.key] == null)

  return (
    <PanelShell title="Medical ASR Accuracy">
      {allNull ? (
        <Unavailable />
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left text-xs font-medium text-gray-400 uppercase pb-2">
                Category
              </th>
              <th className="text-left text-xs font-medium text-gray-400 uppercase pb-2">
                WER
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {WER_ROWS.map((row) => (
              <tr key={row.key}>
                <td className="py-2 text-gray-600">{row.label}</td>
                <td className="py-2">
                  <WerDot value={metrics[row.key]} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </PanelShell>
  )
}

// ---------------------------------------------------------------------------
// Panel 3 — Hallucination Report
// ---------------------------------------------------------------------------

const HALLUCINATION_DISCLAIMER =
  'Grounding uses sentence-level cosine similarity (threshold 0.45). ' +
  'Not a clinical validation tool.'

function HallucinationPanel({ metrics, onHighlightSegment }) {
  if (!metrics) return <PanelSkeleton title="Hallucination Check" />

  const flags = metrics.hallucination_flags
  if (!flags) {
    return <PanelShell title="Hallucination Check"><Unavailable /></PanelShell>
  }

  return (
    <PanelShell title="Hallucination Check">
      {flags.length === 0 ? (
        <p className="text-sm text-green-600 font-medium">
          All claims grounded in transcript ✓
        </p>
      ) : (
        <ul className="space-y-1">
          {flags.map((flag, i) => (
            <li
              key={i}
              onClick={() => onHighlightSegment?.(i)}
              className={[
                'flex items-start gap-2 rounded px-2 py-1.5 text-sm transition-colors',
                onHighlightSegment ? 'cursor-pointer' : '',
                flag.grounded ? 'hover:bg-green-50' : 'hover:bg-red-50',
              ].join(' ')}
            >
              <span
                className={`shrink-0 font-bold text-base leading-none mt-0.5 ${
                  flag.grounded ? 'text-green-500' : 'text-red-500'
                }`}
              >
                {flag.grounded ? '✓' : '✗'}
              </span>
              <span className="flex-1 text-gray-700 leading-snug">{flag.claim}</span>
              <span className="shrink-0 text-xs tabular-nums text-gray-400 mt-0.5">
                {(flag.max_similarity * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="text-xs text-gray-400 italic mt-3 leading-snug">
        {HALLUCINATION_DISCLAIMER}
      </p>
    </PanelShell>
  )
}

// ---------------------------------------------------------------------------
// Panel 4 — PDQI Proxy Radar
// ---------------------------------------------------------------------------

const PDQI_AXES = [
  { key: 'accuracy',      label: 'Accuracy'      },
  { key: 'completeness',  label: 'Completeness'  },
  { key: 'organization',  label: 'Organization'  },
  { key: 'conciseness',   label: 'Conciseness'   },
  { key: 'attribution',   label: 'Attribution'   },
]

const PDQI_DISCLAIMER =
  'LLM-proxy rubric adapted from PDQI-9 (Stetson et al., 2012). ' +
  'Not validated against human clinical raters. Use for relative comparison only.'

function PdqiPanel({ metrics }) {
  if (!metrics) return <PanelSkeleton title="Note Quality (PDQI-9 Proxy)" />

  const { pdqi_scores, pdqi_mean } = metrics
  if (!pdqi_scores) {
    return (
      <PanelShell title="Note Quality (PDQI-9 Proxy)"><Unavailable /></PanelShell>
    )
  }

  const data = PDQI_AXES.map(({ key, label }) => ({
    subject: label,
    value:   pdqi_scores[key] ?? 0,
  }))

  return (
    <PanelShell title="Note Quality (PDQI-9 Proxy)">
      <div className="relative">
        <ResponsiveContainer width="100%" height={170}>
          <RadarChart data={data} margin={{ top: 10, right: 24, bottom: 10, left: 24 }}>
            <PolarGrid stroke="#e5e7eb" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fontSize: 10, fill: '#6b7280' }}
            />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 5]}
              tick={false}
              axisLine={false}
            />
            <Radar
              dataKey="value"
              fill="#bfdbfe"
              fillOpacity={0.7}
              stroke="#3b82f6"
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>

        {pdqi_mean != null && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-2xl font-bold text-blue-600 tabular-nums">
              {pdqi_mean.toFixed(1)}
            </span>
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 italic mt-1 text-center leading-snug">
        {PDQI_DISCLAIMER}
      </p>
    </PanelShell>
  )
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function EvalDashboard({ metrics, pipeline, onHighlightSegment }) {
  return (
    <div className="grid grid-cols-2 gap-3 h-full">
      <LatencyPanel metrics={metrics} />
      <WerPanel     metrics={metrics} />
      <HallucinationPanel
        metrics={metrics}
        onHighlightSegment={onHighlightSegment}
      />
      <PdqiPanel metrics={metrics} />
    </div>
  )
}
