// Props:
//   soapNote:          {S, O, A, P, chief_complaint, icd10_suggestions} | null
//   hallucinationFlags: list[{claim, grounded, max_similarity, best_source_text, best_source_speaker}]
//   encounterId:       string
//   onSaved:           () => void

import { useState, useEffect, useRef, useCallback } from 'react'
import { Check, Loader2, X, AlertTriangle } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const SPEAKER_LABEL = { SPEAKER_00: 'Physician', SPEAKER_01: 'Patient' }

const SECTIONS = [
  { key: 'S', label: 'Subjective',  showFlags: false },
  { key: 'O', label: 'Objective',   showFlags: false },
  { key: 'A', label: 'Assessment',  showFlags: true  },
  { key: 'P', label: 'Plan',        showFlags: true  },
]

const EMPTY_VALUES = { S: '', O: '', A: '', P: '' }
const EMPTY_DIRTY  = { S: false, O: false, A: false, P: false }

// ---------------------------------------------------------------------------
// Auto-resizing textarea hook
// ---------------------------------------------------------------------------

function useAutoResize(value) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [value])
  return ref
}

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------

function SkeletonBlock() {
  return (
    <div className="flex flex-col gap-2">
      <div className="h-3 w-20 bg-gray-100 rounded animate-pulse" />
      <div className="h-24 bg-gray-100 rounded-lg animate-pulse" />
    </div>
  )
}

function SkeletonLoader() {
  return (
    <div className="grid grid-cols-2 gap-3 p-4">
      <SkeletonBlock />
      <SkeletonBlock />
      <SkeletonBlock />
      <SkeletonBlock />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Hallucination drawer
// ---------------------------------------------------------------------------

function HallucinationDrawer({ flags, onClose }) {
  const ungrounded = flags.filter((f) => !f.grounded)

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/25"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed right-0 top-0 h-full w-96 z-50 bg-white shadow-2xl flex flex-col">
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div>
            <h3 className="font-semibold text-gray-900 text-sm">Ungrounded Claims</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              {ungrounded.length} claim{ungrounded.length !== 1 ? 's' : ''} not found in transcript
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 text-gray-500 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {ungrounded.length === 0 ? (
            <p className="text-sm text-gray-400 text-center mt-6">All claims are grounded ✓</p>
          ) : (
            ungrounded.map((flag, i) => (
              <div
                key={i}
                className="border border-orange-200 rounded-lg p-3 bg-orange-50 space-y-1.5"
              >
                <p className="text-sm font-medium text-gray-900">"{flag.claim}"</p>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-orange-700 font-medium">
                    {(flag.max_similarity * 100).toFixed(0)}% similarity
                  </span>
                  <span className="text-gray-300">·</span>
                  <span className="text-xs text-gray-500">
                    {SPEAKER_LABEL[flag.best_source_speaker] ?? flag.best_source_speaker}
                  </span>
                </div>
                {flag.best_source_text && (
                  <p className="text-xs text-gray-600 italic border-l-2 border-orange-300 pl-2">
                    "{flag.best_source_text}"
                  </p>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Single section textarea
// ---------------------------------------------------------------------------

function SectionEditor({ sectionKey, label, value, onChange, flagCount, onFlagBadgeClick }) {
  const textareaRef = useAutoResize(value)

  return (
    <div className="flex flex-col gap-1.5">
      {/* Label row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-bold text-gray-400 uppercase">{sectionKey}</span>
          <span className="text-xs text-gray-500">{label}</span>
          {flagCount > 0 && (
            <button
              onClick={onFlagBadgeClick}
              className="flex items-center gap-0.5 bg-orange-100 hover:bg-orange-200
                         text-orange-700 text-xs font-semibold px-1.5 py-0.5 rounded-full
                         transition-colors"
              title={`${flagCount} ungrounded claim${flagCount !== 1 ? 's' : ''}`}
            >
              <AlertTriangle className="w-3 h-3" />
              {flagCount}
            </button>
          )}
        </div>
        <span className="text-xs text-gray-400 tabular-nums">{value.length}</span>
      </div>

      {/* Textarea */}
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(sectionKey, e.target.value)}
        rows={4}
        className="w-full resize-none rounded-lg border border-gray-200 px-3 py-2
                   text-sm text-gray-800 leading-relaxed
                   focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-transparent
                   overflow-hidden"
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function SOAPEditor({ soapNote, hallucinationFlags = [], encounterId, onSaved }) {
  const [values,      setValues]      = useState(EMPTY_VALUES)
  const [original,   setOriginal]    = useState(EMPTY_VALUES)
  const [dirty,      setDirty]       = useState(EMPTY_DIRTY)
  const [icdCodes,   setIcdCodes]    = useState([])
  const [dismissed,  setDismissed]   = useState(new Set())
  const [drawerOpen, setDrawerOpen]  = useState(false)
  const [saving,     setSaving]      = useState(false)
  const [savedFlash, setSavedFlash]  = useState(false)
  const [saveError,  setSaveError]   = useState(null)

  // Sync incoming soapNote
  useEffect(() => {
    if (!soapNote) return
    const next = {
      S: soapNote.S ?? '',
      O: soapNote.O ?? '',
      A: soapNote.A ?? '',
      P: soapNote.P ?? '',
    }
    setValues(next)
    setOriginal(next)
    setDirty(EMPTY_DIRTY)
    setIcdCodes(soapNote.icd10_suggestions ?? [])
    setDismissed(new Set())
    setSaveError(null)
  }, [soapNote])

  const handleChange = useCallback((key, val) => {
    setValues((prev) => ({ ...prev, [key]: val }))
    setDirty((prev) => ({ ...prev, [key]: val !== original[key] }))
  }, [original])

  const hasDirty = Object.values(dirty).some(Boolean)
  const ungroundedCount = hallucinationFlags.filter((f) => !f.grounded).length

  const save = async () => {
    if (!encounterId || !hasDirty) return
    setSaving(true)
    setSaveError(null)

    const toSave = SECTIONS.map((s) => s.key).filter((k) => dirty[k])

    try {
      await Promise.all(
        toSave.map((section) =>
          fetch(`${API_BASE}/encounter/${encounterId}/edit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ section, edited_text: values[section] }),
          }).then((r) => {
            if (!r.ok) throw new Error(`Save failed for section ${section} (${r.status})`)
          })
        )
      )
      setOriginal({ ...values })
      setDirty(EMPTY_DIRTY)
      onSaved?.()
      setSavedFlash(true)
      setTimeout(() => setSavedFlash(false), 2000)
    } catch (err) {
      setSaveError(err.message)
    } finally {
      setSaving(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <>
      <div className="bg-white rounded-lg border border-gray-200 flex flex-col">
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
              SOAP Note
            </h2>
            {soapNote?.chief_complaint && (
              <span className="text-xs text-gray-400 italic">
                CC: {soapNote.chief_complaint}
              </span>
            )}
          </div>

          {/* Save controls */}
          <div className="flex items-center gap-2">
            {saveError && (
              <span className="text-xs text-red-600">{saveError}</span>
            )}
            {savedFlash && (
              <span className="flex items-center gap-1 text-xs text-green-600 font-medium">
                <Check className="w-3.5 h-3.5" />
                Saved
              </span>
            )}
            <button
              onClick={save}
              disabled={!hasDirty || saving || !encounterId}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-500 hover:bg-blue-600
                         text-white text-xs font-medium disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors"
            >
              {saving && <Loader2 className="w-3 h-3 animate-spin" />}
              Save
            </button>
          </div>
        </div>

        {/* Body */}
        {!soapNote ? (
          <SkeletonLoader />
        ) : (
          <div className="p-4 flex flex-col gap-4">
            {/* 2×2 grid */}
            <div className="grid grid-cols-2 gap-3">
              {SECTIONS.map((sec) => (
                <SectionEditor
                  key={sec.key}
                  sectionKey={sec.key}
                  label={sec.label}
                  value={values[sec.key]}
                  onChange={handleChange}
                  flagCount={sec.showFlags ? ungroundedCount : 0}
                  onFlagBadgeClick={() => setDrawerOpen(true)}
                />
              ))}
            </div>

            {/* ICD-10 chips */}
            {icdCodes.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase mb-1.5">ICD-10</p>
                <div className="flex flex-wrap gap-1.5">
                  {icdCodes.map((code, i) =>
                    dismissed.has(i) ? null : (
                      <span
                        key={i}
                        className="flex items-center gap-1 text-xs bg-gray-100 text-gray-700
                                   rounded-full px-2.5 py-1"
                      >
                        {code}
                        <button
                          onClick={() => setDismissed((prev) => new Set([...prev, i]))}
                          className="ml-0.5 text-gray-400 hover:text-gray-700 transition-colors"
                          aria-label="Dismiss"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hallucination drawer — rendered outside card to avoid clipping */}
      {drawerOpen && (
        <HallucinationDrawer
          flags={hallucinationFlags}
          onClose={() => setDrawerOpen(false)}
        />
      )}
    </>
  )
}
