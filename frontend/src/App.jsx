import { useState, useCallback } from 'react'
import AudioCapture from './components/AudioCapture'
import TranscriptPanel from './components/TranscriptPanel'
import SOAPEditor from './components/SOAPEditor'
import EvalDashboard from './components/EvalDashboard'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [encounterId, setEncounterId]           = useState(null)
  const [encounterData, setEncounterData]       = useState(null)
  const [highlightedSegments, setHighlighted]   = useState([])

  const fetchEncounterData = useCallback(async (id) => {
    try {
      const res = await fetch(`${API_BASE}/encounter/${id}`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setEncounterData(await res.json())
    } catch (err) {
      console.error('Failed to load encounter data:', err)
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shrink-0">
        <h1 className="text-base font-semibold text-gray-900 tracking-tight">
          Ambient Clinical Scribe
        </h1>
        <span className="bg-red-600 text-white text-xs font-bold px-3 py-1 rounded select-none">
          SYNTHETIC DATA ONLY — NOT FOR CLINICAL USE
        </span>
      </header>

      {/* Two-column grid: 40 / 60 */}
      <main className="flex-1 grid grid-cols-[40%_1fr] gap-3 p-3 min-h-0">
        {/* Left column */}
        <div className="flex flex-col gap-3 min-h-0">
          <AudioCapture
            onEncounterStart={(id) => setEncounterId(id)}
            onEncounterComplete={(id) => fetchEncounterData(id)}
          />
          <div className="flex-1 min-h-0">
            <TranscriptPanel
              segments={encounterData?.diarized_segments ?? []}
              highlightedSegments={highlightedSegments}
            />
          </div>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-3 min-h-0">
          <SOAPEditor
            soapNote={
              encounterData?.soap_note?.note
                ? {
                    S: encounterData.soap_note.note.subjective ?? '',
                    O: encounterData.soap_note.note.objective ?? '',
                    A: encounterData.soap_note.note.assessment ?? '',
                    P: encounterData.soap_note.note.plan ?? '',
                    chief_complaint: encounterData.soap_note.note.chief_complaint ?? '',
                    icd10_suggestions: encounterData.soap_note.note.icd10_suggestions ?? [],
                  }
                : null
            }
            hallucinationFlags={encounterData?.metrics?.hallucination_flags ?? []}
            encounterId={encounterId}
            onSaved={() => fetchEncounterData(encounterId)}
          />
          <div className="flex-1 min-h-0">
            <EvalDashboard
              metrics={encounterData?.metrics ?? null}
              pipeline={encounterData?.pipeline ?? null}
              onHighlightSegment={(i) => setHighlighted([i])}
            />
          </div>
        </div>
      </main>
    </div>
  )
}
