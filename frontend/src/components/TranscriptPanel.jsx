// Props:
//   segments: list[{start, end, text, speaker}] | null
//   highlightedSegments: list[number]  — indices to highlight (hallucination evidence)

const SPEAKER = {
  SPEAKER_00: { label: 'Physician', side: 'right', bubble: 'bg-blue-100 text-blue-900',  label_cls: 'text-blue-600' },
  SPEAKER_01: { label: 'Patient',   side: 'left',  bubble: 'bg-gray-100 text-gray-900',  label_cls: 'text-gray-500' },
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

function SkeletonBubble({ side, width }) {
  return (
    <div className={`flex ${side === 'right' ? 'justify-end' : 'justify-start'}`}>
      <div className={`${width} h-14 rounded-2xl bg-gray-100 animate-pulse`} />
    </div>
  )
}

function SkeletonLoader() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <SkeletonBubble side="right" width="w-2/3" />
      <SkeletonBubble side="left"  width="w-3/4" />
      <SkeletonBubble side="right" width="w-1/2" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Single bubble
// ---------------------------------------------------------------------------

function Bubble({ seg, index, highlighted }) {
  const meta   = SPEAKER[seg.speaker] ?? { label: seg.speaker || '?', side: 'left', bubble: 'bg-gray-100 text-gray-800', label_cls: 'text-gray-400' }
  const isRight = meta.side === 'right'

  return (
    <div className={`flex flex-col gap-0.5 ${isRight ? 'items-end' : 'items-start'}`}>
      {/* Timestamp + speaker label */}
      <div className={`flex items-center gap-2 ${isRight ? 'flex-row-reverse' : ''}`}>
        <span className={`text-xs font-medium ${meta.label_cls}`}>{meta.label}</span>
        <span className="text-xs text-gray-400 tabular-nums">[{formatTime(seg.start)}]</span>
      </div>

      {/* Bubble */}
      <div
        className={[
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
          meta.bubble,
          highlighted
            ? 'border-l-4 border-red-400'
            : '',
        ].join(' ')}
      >
        {seg.text}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function TranscriptPanel({ segments = [], highlightedSegments = [] }) {
  if (!segments || segments.length === 0) {
    return (
      <div className="p-4 text-gray-400 text-sm">
        Transcript will appear here after processing.
      </div>
    )
  }

  return (
    <div className="h-full bg-white rounded-lg border border-gray-200 flex flex-col">
      <div className="px-4 py-3 border-b border-gray-100 shrink-0">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Transcript
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        {segments === null ? (
          <SkeletonLoader />
        ) : segments.length === 0 ? (
          <p className="text-sm text-gray-400 text-center mt-10 px-4">
            No transcript available
          </p>
        ) : (
          <div className="flex flex-col gap-3 p-4">
            {segments.map((seg, i) => (
              <Bubble
                key={i}
                seg={seg}
                index={i}
                highlighted={highlightedSegments.includes(i)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
