import { useState, useRef, useCallback, useEffect } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Circle,
  Loader2,
  Mic,
  Square,
  Upload,
  XCircle,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const WS_BASE = API_BASE.replace(/^http/, 'ws')
const MAX_WS_RECONNECTS = 2

const STAGES = [
  { key: 'audio_upload',    label: 'Audio Upload' },
  { key: 'transcription',   label: 'Transcription' },
  { key: 'soap_generation', label: 'SOAP Generation' },
  { key: 'evaluation',      label: 'Evaluation' },
]

const INITIAL_STATUSES = Object.fromEntries(STAGES.map((s) => [s.key, 'pending']))

const SPECIALTIES = [
  { value: 'primary_care', label: 'Primary Care' },
  { value: 'cardiology',   label: 'Cardiology' },
]

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StageIcon({ status }) {
  switch (status) {
    case 'running':  return <Loader2    className="w-4 h-4 text-blue-500 animate-spin" />
    case 'complete': return <CheckCircle2 className="w-4 h-4 text-green-500" />
    case 'failed':   return <XCircle    className="w-4 h-4 text-red-500" />
    default:         return <Circle     className="w-4 h-4 text-gray-300" />
  }
}

function stageLabelClass(status) {
  switch (status) {
    case 'running':  return 'text-blue-600 font-medium'
    case 'complete': return 'text-green-700'
    case 'failed':   return 'text-red-600 font-medium'
    default:         return 'text-gray-400'
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AudioCapture({ onEncounterStart, onEncounterComplete }) {
  const [specialty, setSpecialty]         = useState('primary_care')
  const [isRecording, setIsRecording]     = useState(false)
  const [isProcessing, setIsProcessing]   = useState(false)
  const [stageStatuses, setStageStatuses] = useState(INITIAL_STATUSES)
  const [uploadError, setUploadError]     = useState(null)
  const [wsError, setWsError]             = useState(null)

  const mediaRecorderRef      = useRef(null)
  const audioChunksRef        = useRef([])
  const wsRef                 = useRef(null)
  const reconnectCountRef     = useRef(0)
  const fileInputRef          = useRef(null)

  // Close any open WS on unmount
  useEffect(() => () => wsRef.current?.close(), [])

  // ----- WebSocket -----

  const connectWS = useCallback((encounterId) => {
    if (reconnectCountRef.current >= MAX_WS_RECONNECTS) {
      setWsError(
        `Pipeline updates unavailable after ${MAX_WS_RECONNECTS} reconnect attempts. ` +
        'Refresh the page to check encounter status.'
      )
      setIsProcessing(false)
      return
    }

    const ws = new WebSocket(`${WS_BASE}/ws/${encounterId}`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      let msg
      try { msg = JSON.parse(e.data) } catch { return }

      const { stage, status } = msg
      if (!stage || !status) return

      setStageStatuses((prev) => ({ ...prev, [stage]: status }))

      if (stage === 'evaluation' && status === 'complete') {
        setIsProcessing(false)
        onEncounterComplete?.(encounterId)
      }
      if (status === 'failed') {
        setIsProcessing(false)
      }
    }

    ws.onclose = () => {
      if (reconnectCountRef.current < MAX_WS_RECONNECTS) {
        reconnectCountRef.current += 1
        setTimeout(() => connectWS(encounterId), 1500)
      } else {
        setWsError('WebSocket connection lost. Pipeline may still be running — refresh to check.')
        setIsProcessing(false)
      }
    }

    ws.onerror = () => ws.close()
  }, [onEncounterComplete])

  // ----- Upload -----

  const uploadAudio = useCallback(async (fileOrBlob, filename) => {
    setUploadError(null)
    setWsError(null)
    setIsProcessing(true)
    reconnectCountRef.current = 0
    setStageStatuses({ ...INITIAL_STATUSES, audio_upload: 'running' })

    const formData = new FormData()
    formData.append('audio_file', fileOrBlob, filename)
    formData.append('specialty', specialty)

    try {
      const res = await fetch(`${API_BASE}/encounter/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const body = await res.text().catch(() => res.statusText)
        throw new Error(`Upload failed (${res.status}): ${body}`)
      }

      const { encounter_id } = await res.json()
      console.log("encounter_id received:", encounter_id)
      const wsUrl = `${WS_BASE}/ws/${encounter_id}`
      console.log("connecting to:", wsUrl)
      onEncounterStart?.(encounter_id)

      setStageStatuses((prev) => ({
        ...prev,
        audio_upload: 'complete',
        transcription: 'running',
      }))

      connectWS(encounter_id)
    } catch (err) {
      setStageStatuses((prev) => ({ ...prev, audio_upload: 'failed' }))
      setUploadError(err.message)
      setIsProcessing(false)
    }
  }, [specialty, onEncounterStart, connectWS])

  // ----- Recording -----

  const startRecording = useCallback(async () => {
    setUploadError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []

      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        uploadAudio(blob, 'recording.webm')
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch (err) {
      setUploadError(`Microphone access denied: ${err.message}`)
    }
  }, [uploadAudio])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }, [])

  // ----- File input -----

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return
    uploadAudio(file, file.name)
    e.target.value = ''
  }, [uploadAudio])

  const controlsDisabled = isRecording || isProcessing

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 flex flex-col gap-4">
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
        Audio Capture
      </h2>

      {/* Specialty selector */}
      <div className="flex items-center gap-3">
        <label className="text-sm text-gray-600 shrink-0">Specialty</label>
        <select
          value={specialty}
          onChange={(e) => setSpecialty(e.target.value)}
          disabled={controlsDisabled}
          className="flex-1 text-sm border border-gray-300 rounded px-2 py-1.5
                     disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          {SPECIALTIES.map((s) => (
            <option key={s.value} value={s.value}>{s.label}</option>
          ))}
        </select>
      </div>

      {/* Record / Stop / Upload */}
      <div className="flex gap-2">
        {isRecording ? (
          <button
            onClick={stopRecording}
            className="flex items-center gap-2 px-4 py-2 rounded bg-gray-800 hover:bg-gray-900
                       text-white text-sm font-medium transition-colors"
          >
            <Square className="w-4 h-4 fill-current" />
            Stop Recording
          </button>
        ) : (
          <button
            onClick={startRecording}
            disabled={controlsDisabled}
            className="flex items-center gap-2 px-4 py-2 rounded bg-red-500 hover:bg-red-600
                       text-white text-sm font-medium disabled:opacity-40
                       disabled:cursor-not-allowed transition-colors"
          >
            <Mic className="w-4 h-4" />
            Record
          </button>
        )}

        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={controlsDisabled}
          className="flex items-center gap-2 px-4 py-2 rounded border border-gray-300
                     hover:bg-gray-50 text-gray-700 text-sm font-medium
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload File
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.m4a"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {/* Pipeline stepper */}
      <div className="border-t border-gray-100 pt-3">
        <p className="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
          Pipeline
        </p>
        <div className="flex items-center gap-1 flex-wrap">
          {STAGES.map((stage, i) => (
            <div key={stage.key} className="flex items-center gap-1">
              <div className="flex items-center gap-1.5">
                <StageIcon status={stageStatuses[stage.key]} />
                <span className={`text-xs ${stageLabelClass(stageStatuses[stage.key])}`}>
                  {stage.label}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <ChevronRight className="w-3 h-3 text-gray-300 mx-0.5 shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Error banners */}
      {(uploadError || wsError) && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded
                        px-3 py-2 text-sm text-red-700">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{uploadError ?? wsError}</span>
        </div>
      )}
    </div>
  )
}
