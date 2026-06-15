export default function EvalDashboard({ encounterData }) {
  const metrics = encounterData?.metrics ?? null

  return (
    <div className="h-full bg-white rounded-lg border border-gray-200 flex flex-col">
      <div className="px-4 py-3 border-b border-gray-100">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Evaluation
        </h2>
      </div>

      {!metrics ? (
        <p className="text-sm text-gray-400 text-center p-8">
          Evaluation metrics will appear after processing
        </p>
      ) : (
        <div className="p-4 space-y-3 overflow-y-auto flex-1">
          {/* Latency */}
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Latency</p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              <span className="text-gray-500">ASR</span>
              <span className="text-gray-800 text-right">{metrics.asr_ms?.toFixed(0)} ms</span>
              <span className="text-gray-500">Diarization</span>
              <span className="text-gray-800 text-right">{metrics.diarization_ms?.toFixed(0)} ms</span>
              <span className="text-gray-500">Note generation</span>
              <span className="text-gray-800 text-right">{metrics.note_gen_ms?.toFixed(0)} ms</span>
              <span className="text-gray-500 font-medium">Total</span>
              <span className="text-gray-900 font-medium text-right">{metrics.total_ms?.toFixed(0)} ms</span>
            </div>
          </div>

          {/* PDQI */}
          {metrics.pdqi_mean != null && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase mb-1">
                PDQI-9 Proxy
              </p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${(metrics.pdqi_mean / 5) * 100}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-800">
                  {metrics.pdqi_mean?.toFixed(1)} / 5
                </span>
              </div>
            </div>
          )}

          {/* Hallucination */}
          {metrics.hallucination_count != null && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase mb-1">
                Hallucination Check
              </p>
              <p className="text-sm text-gray-800">
                {metrics.hallucination_count === 0 ? (
                  <span className="text-green-600">All claims grounded ✓</span>
                ) : (
                  <span className="text-red-600">
                    {metrics.hallucination_count} ungrounded claim
                    {metrics.hallucination_count !== 1 ? 's' : ''}
                  </span>
                )}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
