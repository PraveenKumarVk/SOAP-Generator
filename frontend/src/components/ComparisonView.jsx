// Props:
//   data:    object keyed by encounter_id (response from GET /compare)
//   onClose: () => void

import { X, Download } from 'lucide-react'

const LFM_PIPELINES = new Set(['lfm', 'hybrid'])

// ---------------------------------------------------------------------------
// Badges
// ---------------------------------------------------------------------------

function PipelineBadge({ pipeline }) {
  if (pipeline === 'lfm') {
    return (
      <span className="inline-flex items-center bg-purple-100 text-purple-700
                       text-xs font-medium px-2 py-0.5 rounded-full">
        LFM 2.5
      </span>
    )
  }
  if (pipeline === 'hybrid') {
    return (
      <span className="inline-flex items-center bg-indigo-100 text-indigo-700
                       text-xs font-medium px-2 py-0.5 rounded-full">
        Hybrid
      </span>
    )
  }
  return (
    <span className="inline-flex items-center bg-blue-100 text-blue-700
                     text-xs font-medium px-2 py-0.5 rounded-full">
      WhisperX
    </span>
  )
}

function PendingBadge() {
  return (
    <span className="inline-flex items-center bg-gray-100 text-gray-400
                     text-xs italic px-2 py-0.5 rounded-full">
      Pending integration
    </span>
  )
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

function exportCsv(rows) {
  const headers = [
    'ID', 'Pipeline', 'Specialty', 'Status',
    'ASR (ms)', 'Diarization (ms)', 'SOAP Gen (ms)', 'Total (ms)',
    'WER Medications (%)', 'WER Symptoms (%)',
    'Hallucinations', 'PDQI Mean',
  ]
  const csvRows = rows.map((r) => {
    const m = r.metrics ?? {}
    return [
      r.id,
      r.pipeline ?? '',
      r.specialty ?? '',
      r.status ?? '',
      m.asr_ms?.toFixed(0) ?? '',
      m.diarization_ms?.toFixed(0) ?? '',
      m.note_gen_ms?.toFixed(0) ?? '',
      m.total_ms?.toFixed(0) ?? '',
      m.medical_wer_medications != null ? (m.medical_wer_medications * 100).toFixed(1) : '',
      m.medical_wer_symptoms != null    ? (m.medical_wer_symptoms * 100).toFixed(1) : '',
      m.hallucination_count ?? '',
      m.pdqi_mean?.toFixed(2) ?? '',
    ].map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')
  })

  const csv = [headers.join(','), ...csvRows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `encounter_comparison_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// Table row
// ---------------------------------------------------------------------------

function EncounterRow({ r }) {
  const m = r.metrics ?? {}
  const isLfm = LFM_PIPELINES.has(r.pipeline)

  const werValues = [m.medical_wer_medications, m.medical_wer_symptoms].filter((v) => v != null)
  const werDisplay = werValues.length
    ? werValues.map((v) => (v * 100).toFixed(1) + '%').join(' / ')
    : null

  return (
    <tr className="hover:bg-gray-50 transition-colors">
      {/* Encounter */}
      <td className="py-3 pr-4">
        <div className="font-mono text-xs text-gray-700 truncate max-w-[160px]">
          {r.id}
        </div>
        <div className="text-xs text-gray-400 mt-0.5 truncate max-w-[160px]">
          {r.audio_filename ?? '—'}
        </div>
      </td>

      {/* Pipeline */}
      <td className="py-3 pr-4">
        <PipelineBadge pipeline={r.pipeline} />
      </td>

      {/* Specialty */}
      <td className="py-3 pr-4 text-sm text-gray-600 capitalize">
        {r.specialty?.replace('_', ' ') ?? '—'}
      </td>

      {/* Total latency */}
      <td className="py-3 pr-4 text-right tabular-nums text-sm text-gray-800">
        {m.total_ms != null ? `${m.total_ms.toFixed(0)} ms` : '—'}
      </td>

      {/* Med WER */}
      <td className="py-3 pr-4 text-right tabular-nums text-sm">
        {isLfm && werDisplay == null ? (
          <PendingBadge />
        ) : werDisplay != null ? (
          <span className="text-gray-800">{werDisplay}</span>
        ) : (
          <span className="text-gray-400">N/A</span>
        )}
      </td>

      {/* Hallucinations */}
      <td className="py-3 pr-4 text-right tabular-nums text-sm">
        {isLfm && m.hallucination_count == null ? (
          <PendingBadge />
        ) : m.hallucination_count != null ? (
          <span className={m.hallucination_count === 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
            {m.hallucination_count}
          </span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
      </td>

      {/* PDQI */}
      <td className="py-3 text-right tabular-nums text-sm">
        {isLfm && m.pdqi_mean == null ? (
          <PendingBadge />
        ) : m.pdqi_mean != null ? (
          <span className="text-gray-800">{m.pdqi_mean.toFixed(2)}</span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
      </td>
    </tr>
  )
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function ComparisonView({ data, onClose }) {
  const rows    = Object.values(data).filter((r) => !r.error)
  const errored = Object.values(data).filter((r) => r.error)
  const hasLfm  = rows.some((r) => LFM_PIPELINES.has(r.pipeline))

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40 bg-black/40" onClick={onClose} />

      {/* Panel */}
      <div className="fixed inset-4 z-50 bg-white rounded-xl shadow-2xl flex flex-col overflow-hidden">

        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Pipeline Comparison</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {rows.length} encounter{rows.length !== 1 ? 's' : ''}
              {!hasLfm && (
                <span className="ml-2 italic text-gray-400">
                  — LFM cells show placeholder until LFM/Hybrid encounters are added
                </span>
              )}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => exportCsv(rows)}
              disabled={rows.length === 0}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-gray-300
                         hover:bg-gray-50 text-sm text-gray-700 font-medium
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded hover:bg-gray-100 text-gray-500 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto p-6">
          {errored.length > 0 && (
            <p className="text-xs text-red-600 mb-3">
              {errored.length} encounter(s) not found:{' '}
              {errored.map((r) => r.id).join(', ')}
            </p>
          )}

          {rows.length === 0 ? (
            <p className="text-sm text-gray-400 text-center mt-10">
              No valid encounters to compare.
            </p>
          ) : (
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  <th className="text-left pb-3 pr-4 border-b border-gray-200">Encounter</th>
                  <th className="text-left pb-3 pr-4 border-b border-gray-200">Pipeline</th>
                  <th className="text-left pb-3 pr-4 border-b border-gray-200">Specialty</th>
                  <th className="text-right pb-3 pr-4 border-b border-gray-200">Total Latency</th>
                  <th className="text-right pb-3 pr-4 border-b border-gray-200">Med WER</th>
                  <th className="text-right pb-3 pr-4 border-b border-gray-200">Hallucinations</th>
                  <th className="text-right pb-3 border-b border-gray-200">PDQI Mean</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {rows.map((r) => (
                  <EncounterRow key={r.id} r={r} />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  )
}
