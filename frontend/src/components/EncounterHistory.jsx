// Props:
//   onCompare: (data: object) => void  — called with raw /compare response
//   onClose:   () => void

import { useState, useEffect, useCallback } from 'react'
import { X, RefreshCw, GitCompare, Loader2 } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusDot({ status }) {
  const cls = {
    complete:   'bg-green-500',
    processing: 'bg-blue-400 animate-pulse',
    failed:     'bg-red-500',
  }[status] ?? 'bg-gray-300'
  return <span className={`w-1.5 h-1.5 rounded-full shrink-0 mt-1 ${cls}`} />
}

function PipelineTag({ pipeline }) {
  const labels = {
    whisperx:      'WhisperX',
    'openai-whisper': 'Whisper',
    lfm:           'LFM',
    hybrid:        'Hybrid',
  }
  return (
    <span className="text-xs text-gray-400 ml-auto shrink-0">
      {labels[pipeline] ?? pipeline ?? 'whisperx'}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function EncounterHistory({ onCompare, onClose }) {
  const [encounters, setEncounters] = useState([])
  const [selected,   setSelected]   = useState(new Set())
  const [loading,    setLoading]    = useState(false)
  const [comparing,  setComparing]  = useState(false)
  const [error,      setError]      = useState(null)

  const fetchEncounters = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/encounters`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setEncounters(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchEncounters() }, [fetchEncounters])

  const toggleSelect = (id) => {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const handleCompare = async () => {
    if (selected.size < 2) return
    setComparing(true)
    setError(null)
    try {
      const ids = [...selected].join(',')
      const res = await fetch(
        `${API_BASE}/compare?encounter_ids=${encodeURIComponent(ids)}`
      )
      if (!res.ok) throw new Error(`Compare failed (${res.status})`)
      onCompare(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setComparing(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-30 bg-black/20"
        onClick={onClose}
      />

      {/* Slide-in panel */}
      <div className="fixed right-0 top-0 h-full w-80 z-40 bg-white shadow-2xl
                      border-l border-gray-200 flex flex-col">

        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <h2 className="text-sm font-semibold text-gray-900">Encounter History</h2>
          <div className="flex items-center gap-1">
            <button
              onClick={fetchEncounters}
              disabled={loading}
              className="p-1.5 rounded hover:bg-gray-100 text-gray-500
                         disabled:opacity-40 transition-colors"
              title="Refresh"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded hover:bg-gray-100 text-gray-500 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Hint */}
        <p className="px-4 py-2 text-xs text-gray-400 border-b border-gray-100 shrink-0">
          Select 2 or more encounters to compare pipelines.
        </p>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {error && (
            <p className="px-4 py-3 text-xs text-red-600">{error}</p>
          )}

          {loading && encounters.length === 0 && (
            <div className="flex justify-center py-8">
              <Loader2 className="w-5 h-5 text-gray-300 animate-spin" />
            </div>
          )}

          {!loading && encounters.length === 0 && !error && (
            <p className="px-4 py-6 text-xs text-gray-400 text-center">
              No encounters recorded yet.
            </p>
          )}

          {encounters.map((enc) => (
            <label
              key={enc.id}
              className={[
                'flex items-start gap-3 px-4 py-3 cursor-pointer border-b border-gray-50',
                'hover:bg-gray-50 transition-colors select-none',
                selected.has(enc.id) ? 'bg-blue-50' : '',
              ].join(' ')}
            >
              <input
                type="checkbox"
                checked={selected.has(enc.id)}
                onChange={() => toggleSelect(enc.id)}
                className="mt-0.5 shrink-0 accent-blue-500"
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <StatusDot status={enc.status} />
                  <span className="text-xs font-mono text-gray-700 truncate">
                    {enc.id.slice(0, 8)}…
                  </span>
                  <PipelineTag pipeline={enc.pipeline} />
                </div>
                <p className="text-xs text-gray-500 mt-0.5 truncate">
                  {enc.audio_filename ?? '—'}
                </p>
                <p className="text-xs text-gray-400 capitalize">
                  {enc.specialty?.replace('_', ' ') ?? '—'}
                </p>
              </div>
            </label>
          ))}
        </div>

        {/* Compare button */}
        <div className="p-3 border-t border-gray-200 shrink-0">
          <button
            onClick={handleCompare}
            disabled={selected.size < 2 || comparing}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md
                       bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {comparing
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <GitCompare className="w-4 h-4" />
            }
            {comparing
              ? 'Fetching…'
              : selected.size < 2
                ? 'Compare (select 2+)'
                : `Compare ${selected.size} encounters`
            }
          </button>
        </div>
      </div>
    </>
  )
}
