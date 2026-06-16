"""
Ambient Scribe — Gradio interface for HF Spaces deployment.

Tabs:
  1. Live Pipeline   — record / upload audio, run full pipeline, view results
  2. Demo Encounters — load pre-seeded encounters from demo_data/demo.db
  3. About           — architecture, methodology, limitations
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Set CPU defaults BEFORE any backend import triggers dotenv / config load.
# python-dotenv does NOT override env vars that are already set, so these act
# as HF-Spaces-safe defaults that a local .env can override.
# ---------------------------------------------------------------------------
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("COMPUTE_TYPE", "int8")
os.environ.setdefault("WHISPER_MODEL_SIZE", "tiny")
os.environ.setdefault("ASR_ENGINE", "openai-whisper")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from backend import config  # noqa: E402  — triggers dotenv load
from backend.pipeline.asr import WhisperXProvider  # noqa: E402
from backend.pipeline.soap import generate_soap, validate_soap  # noqa: E402

try:
    from backend.pipeline.eval import (  # noqa: E402
        check_hallucinations,
        compute_medical_wer,
        compute_pdqi_proxy,
    )
    _EVAL_AVAILABLE = True
except (ImportError, AttributeError):
    _EVAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parent
_DEMO_DB_PATH = _ROOT / "demo_data" / "demo.db"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_provider: WhisperXProvider | None = None
_provider_error: Exception | None = None


def _get_provider() -> WhisperXProvider:
    global _provider, _provider_error

    if _provider is not None:
        return _provider
    if _provider_error is not None:
        raise _provider_error

    print("[asr] Loading ASR model …", flush=True)
    try:
        _provider = WhisperXProvider()
        print("[asr] ASR model loaded.", flush=True)
        return _provider
    except Exception as exc:
        _provider_error = exc
        print(f"[asr] ASR model failed to load: {exc}", flush=True)
        raise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPECIALTY_MAP = {
    "Primary Care": "primary_care",
    "Cardiology":   "cardiology",
}

_SPEAKER_LABELS = {
    "SPEAKER_00": "Physician",
    "SPEAKER_01": "Patient",
}

_LATENCY_STAGES = ["ASR", "Diarization", "SOAP Gen", "Eval"]
_LATENCY_COLORS = ["#3b82f6", "#8b5cf6", "#22c55e", "#f97316"]

_PDQI_DIMS = ["accuracy", "completeness", "organization", "conciseness", "attribution"]

_CPU_WARNING = (
    "⏱️ **CPU inference: ~90 s for 3-minute audio.** "
    "Transcription runs on the free-tier CPU. Longer recordings may take several minutes."
)

_DISCLAIMER = (
    "⚠️ **FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY.** "
    "All notes and metrics are generated from synthetic data. "
    "This system is NOT validated for clinical use."
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _make_transcript_df(segments: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "Speaker":  _SPEAKER_LABELS.get(s.get("speaker", ""), s.get("speaker") or "Unknown"),
            "Start":    f"{s.get('start', 0):.1f}s",
            "Text":     s.get("text", "").strip(),
        }
        for s in segments
        if s.get("text", "").strip()
    ]
    return pd.DataFrame(rows, columns=["Speaker", "Start", "Text"]) if rows else pd.DataFrame(columns=["Speaker", "Start", "Text"])


def _make_wer_df(med_wer=None, sym_wer=None, proc_wer=None) -> pd.DataFrame:
    def _fmt(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A (no ground truth)"

    return pd.DataFrame([
        {"Category": "Medications", "WER": _fmt(med_wer)},
        {"Category": "Symptoms",    "WER": _fmt(sym_wer)},
        {"Category": "Procedures",  "WER": _fmt(proc_wer)},
    ])


def _make_halluc_df(flags: list) -> pd.DataFrame:
    if not flags:
        return pd.DataFrame(columns=["Grounded", "Claim", "Similarity", "Best Source"])
    rows = []
    for f in flags:
        if isinstance(f, dict):
            grounded, claim = f.get("grounded", True), f.get("claim", "")
            sim, src = f.get("max_similarity", 0), f.get("best_source_text", "")
        else:
            grounded, claim = f.grounded, f.claim
            sim, src = f.max_similarity, f.best_source_text
        rows.append({
            "Grounded":   "✓" if grounded else "✗",
            "Claim":      claim,
            "Similarity": f"{sim * 100:.0f}%",
            "Best Source": (src or "")[:100],
        })
    return pd.DataFrame(rows)


def _make_latency_fig(asr_ms: float, diar_ms: float, note_ms: float,
                      eval_ms: float, total_ms: float) -> go.Figure:
    values = [asr_ms, diar_ms, note_ms, eval_ms]
    fig = go.Figure()
    for stage, val, color in zip(_LATENCY_STAGES, values, _LATENCY_COLORS):
        fig.add_trace(go.Bar(
            name=stage,
            x=[round(val)],
            y=["Pipeline"],
            orientation="h",
            marker_color=color,
            text=f"{val:.0f}ms",
            textposition="inside",
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Pipeline Latency — Total: {total_ms:,.0f} ms",
        xaxis_title="Milliseconds",
        height=200,
        showlegend=True,
        legend=dict(orientation="h", y=-0.5),
        margin=dict(t=40, b=80, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _make_pdqi_fig(pdqi_scores: dict, pdqi_mean: float) -> go.Figure:
    labels = [d.capitalize() for d in _PDQI_DIMS]
    values = [pdqi_scores.get(d, 0) for d in _PDQI_DIMS]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(191, 219, 254, 0.6)",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9))),
        title=f"PDQI-9 Proxy — Mean: {pdqi_mean:.2f} / 5.0",
        showlegend=False,
        height=340,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def _icd_text(suggestions: list[str]) -> str:
    return "\n".join(suggestions) if suggestions else ""


# ---------------------------------------------------------------------------
# Hallucination HTML helper
# ---------------------------------------------------------------------------

def build_hallucination_html(flags) -> str:
    if not flags:
        return ""
    try:
        if isinstance(flags, str):
            import json
            flags = json.loads(flags)
        ungrounded = [f for f in flags if not f.get("grounded", True)]
        if not ungrounded:
            return ""
        items = "".join(
            f'<li style="margin-bottom:8px">'
            f'<strong>{f["claim"]}</strong><br>'
            f'<span style="color:#888;font-size:0.85em">'
            f'Best match ({f.get("max_similarity",0):.0%}): '
            f'{f.get("best_source_text","")}'
            f'</span></li>'
            for f in ungrounded
        )
        return (
            '<div style="background:#3d1f00;border:1px solid '
            '#ff8c00;border-radius:6px;padding:12px;margin:8px 0">'
            f'<div style="color:#ff8c00;font-weight:600;'
            f'margin-bottom:8px">⚠️ {len(ungrounded)} ungrounded '
            f'claim(s) in Assessment / Plan</div>'
            f'<ul style="margin:0;padding-left:16px;'
            f'color:#e0e0e0">{items}</ul>'
            '<div style="color:#888;font-size:0.75em;'
            'margin-top:8px">Similarity threshold 0.45 — '
            'experimental, not clinically validated</div>'
            '</div>'
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Shared output builder  (live + demo share same 11-field shape)
# ---------------------------------------------------------------------------

def _build_outputs(
    segments:      list[dict],
    soap_dict:     dict,
    asr_ms:        float,
    diar_ms:       float,
    note_ms:       float,
    eval_ms:       float,
    total_ms:      float,
    med_wer:       float | None,
    sym_wer:       float | None,
    proc_wer:      float | None,
    halluc_flags:  list | None,
    pdqi_scores:   dict | None,
    pdqi_mean:     float | None,
) -> tuple:
    note = soap_dict.get("note", soap_dict)  # handle both SOAPResult.model_dump() and flat dict

    transcript_df  = _make_transcript_df(segments)
    subjective     = note.get("subjective", "")
    objective      = note.get("objective", "")
    assessment     = note.get("assessment", "")
    plan           = note.get("plan", "")
    icd_str        = _icd_text(note.get("icd10_suggestions", []))
    latency_fig    = _make_latency_fig(asr_ms, diar_ms, note_ms, eval_ms, total_ms)
    wer_df         = _make_wer_df(med_wer, sym_wer, proc_wer)
    halluc_df      = _make_halluc_df(halluc_flags or [])
    pdqi_fig       = _make_pdqi_fig(pdqi_scores, pdqi_mean) if (pdqi_scores and pdqi_mean is not None) else None

    return (transcript_df, subjective, objective, assessment, plan,
            icd_str, latency_fig, wer_df, halluc_df, pdqi_fig,
            build_hallucination_html(halluc_flags or []))


# ---------------------------------------------------------------------------
# Tab 1 — Live pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str | None, specialty: str, progress=gr.Progress()) -> tuple:
    if audio_path is None:
        raise gr.Error("Please record from the microphone or upload an audio file.")

    config.require_anthropic_api_key()
    specialty_key = _SPECIALTY_MAP.get(specialty, "primary_care")

    # ── Stage 1: Transcription ──────────────────────────────────────────────
    progress(0.05, desc="Running ASR transcription …")
    try:
        tr = _get_provider().transcribe(audio_path)
    except Exception as exc:
        raise gr.Error(f"ASR model failed: {exc}") from exc
    raw_transcript = " ".join(s.text.strip() for s in tr.segments)
    diarized = [s.model_dump() for s in tr.segments]
    asr_ms   = tr.latency.asr_ms + tr.latency.alignment_ms
    diar_ms  = tr.latency.diarization_ms

    # ── Stage 2: SOAP ───────────────────────────────────────────────────────
    progress(0.50, desc="Generating SOAP note …")
    soap = generate_soap(tr.segments, specialty=specialty_key)
    note_ms = soap.generation_ms

    # ── Stage 3: Eval ───────────────────────────────────────────────────────
    med_wer = sym_wer = proc_wer = None
    halluc_flags: list = []
    pdqi_scores: dict | None = None
    pdqi_mean: float | None = None
    t_eval = time.perf_counter()

    if _EVAL_AVAILABLE:
        progress(0.70, desc="Checking hallucinations …")
        try:
            hr = check_hallucinations(soap, tr.segments)
            halluc_flags = [f.model_dump() for f in hr.flags]
        except Exception:
            pass

        progress(0.85, desc="Computing PDQI proxy score …")
        try:
            pr = compute_pdqi_proxy(raw_transcript, soap.model_dump())
            pdqi_scores = pr.scores.model_dump()
            pdqi_mean   = pr.mean_score
        except Exception:
            pass

    eval_ms   = (time.perf_counter() - t_eval) * 1000
    total_ms  = tr.latency.total_ms + soap.generation_ms + eval_ms
    progress(1.0, desc="Complete ✓")

    return _build_outputs(
        segments=diarized,
        soap_dict=soap.model_dump(),
        asr_ms=asr_ms,
        diar_ms=diar_ms,
        note_ms=note_ms,
        eval_ms=eval_ms,
        total_ms=total_ms,
        med_wer=med_wer,
        sym_wer=sym_wer,
        proc_wer=proc_wer,
        halluc_flags=halluc_flags,
        pdqi_scores=pdqi_scores,
        pdqi_mean=pdqi_mean,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Demo encounters
# ---------------------------------------------------------------------------

def _demo_choices() -> list[tuple[str, str]]:
    """Return [(label, encounter_id), …] from demo.db, empty list if not seeded."""
    if not _DEMO_DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(str(_DEMO_DB_PATH))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, audio_filename, specialty FROM encounter "
            "WHERE is_demo=1 AND status='complete' ORDER BY rowid ASC"
        ).fetchall()
        con.close()
        return [
            (
                f"{r['audio_filename']}  ·  {r['specialty'].replace('_', ' ').title()}",
                r["id"],
            )
            for r in rows
        ]
    except Exception:
        return []


def load_demo(encounter_id: str | None) -> tuple:
    if not encounter_id:
        raise gr.Error("Select a demo encounter from the dropdown.")
    if not _DEMO_DB_PATH.exists():
        raise gr.Error(
            "demo_data/demo.db not found. "
            "Run: PYTHONPATH=. .venv/bin/python backend/seed_demo_data.py"
        )

    con = sqlite3.connect(str(_DEMO_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        enc = con.execute("SELECT * FROM encounter WHERE id=?", [encounter_id]).fetchone()
        if enc is None:
            raise gr.Error(f"Encounter {encounter_id!r} not found in demo.db")
        met = con.execute(
            "SELECT * FROM encounter_metrics WHERE encounter_id=?", [encounter_id]
        ).fetchone()
    finally:
        con.close()

    segments  = json.loads(enc["diarized_segments"]) if enc["diarized_segments"] else []
    soap_dict = json.loads(enc["soap_note"])          if enc["soap_note"]          else {}

    asr_ms = diar_ms = note_ms = eval_ms = total_ms = 0.0
    med_wer = sym_wer = proc_wer = None
    halluc_flags: list = []
    pdqi_scores: dict | None = None
    pdqi_mean: float | None = None

    if met:
        asr_ms    = float(met["asr_ms"] or 0)
        diar_ms   = float(met["diarization_ms"] or 0)
        note_ms   = float(met["note_gen_ms"] or 0)
        total_ms  = float(met["total_ms"] or 0)
        eval_ms   = max(0.0, total_ms - asr_ms - diar_ms - note_ms)

        med_wer  = met["medical_wer_medications"]
        sym_wer  = met["medical_wer_symptoms"]
        proc_wer = met["medical_wer_procedures"]

        if met["hallucination_flags"]:
            halluc_flags = json.loads(met["hallucination_flags"])

        if met["pdqi_scores"]:
            pdqi_scores = json.loads(met["pdqi_scores"])
            pdqi_mean   = met["pdqi_mean"]

    return _build_outputs(
        segments=segments,
        soap_dict=soap_dict,
        asr_ms=asr_ms,
        diar_ms=diar_ms,
        note_ms=note_ms,
        eval_ms=eval_ms,
        total_ms=total_ms,
        med_wer=med_wer,
        sym_wer=sym_wer,
        proc_wer=proc_wer,
        halluc_flags=halluc_flags,
        pdqi_scores=pdqi_scores,
        pdqi_mean=pdqi_mean,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_ABOUT_MD = """
## Architecture

```
Audio input
  ├─ VAD (Voice Activity Detection)          WhisperX / openai-whisper
  ├─ ASR Transcription                       Whisper large-v2 (int8, CPU)
  ├─ Word-level Alignment                    wav2vec2 forced alignment
  └─ Speaker Diarization                     pyannote/speaker-diarization-3.1

SOAP Note Generation                         Claude claude-sonnet-4-6 (temperature=0)
  └─ Specialty prompts: Primary Care / Cardiology
  └─ JSON output: S / O / A / P / ICD-10 / raw_claims

Evaluation Suite
  ├─ Medical Entity WER    scispaCy en_ner_bc5cdr_md → jiwer
  ├─ Hallucination Check   sentence-transformers all-MiniLM-L6-v2
  │                        cosine similarity threshold = 0.45
  └─ PDQI-9 Proxy          Claude claude-sonnet-4-6 rubric (5 dimensions, 1–5 scale)
```

## Technical Methodology

**ASR** uses OpenAI Whisper (via `openai-whisper` or `whisperx`) for speech recognition,
followed by `wav2vec2` forced alignment for word-level timestamps.
Speaker diarization is performed by `pyannote.audio 3.0`, which assigns each segment
a `SPEAKER_00` (Physician) or `SPEAKER_01` (Patient) label.

**SOAP generation** sends the diarized transcript to `claude-sonnet-4-6` with a
specialty-specific system prompt and requires structured JSON output with exactly seven
keys. ICD-10 codes are constrained to a maximum of three, grounded in the transcript.

**Medical WER** extracts named entities (CHEMICAL = medications, DISEASE = symptoms)
from both the reference and hypothesis transcripts using `en_ner_bc5cdr_md` and computes
word error rate with `jiwer`. Requires a reference transcript — shown as N/A for live
encounters without ground truth.

**Hallucination detection** encodes each discrete claim from the SOAP Assessment/Plan
and each transcript sentence with `all-MiniLM-L6-v2`, then checks whether the
maximum cosine similarity exceeds 0.45. Claims below this threshold are flagged
as ungrounded.

**PDQI-9 Proxy** prompts `claude-sonnet-4-6` to score the generated note on five
dimensions adapted from Stetson et al. (2012): Accuracy, Completeness, Organization,
Conciseness, and Attribution (1–5 scale each).

## Limitations

- **Not validated for clinical use.** All outputs are from a research prototype.
- Medical WER requires a human-annotated reference transcript; live encounters show N/A.
- PDQI proxy scores are LLM-generated, not validated against human clinical raters.
- Hallucination threshold (0.45) is heuristic; false positives are expected for
  medical paraphrases and abbreviations.
- Speaker diarization assumes exactly 2 speakers and may mis-assign in multi-party recordings.
- CPU inference on free-tier HF Spaces is significantly slower than GPU deployment.

## References

- Stetson PD et al. *The note quality rubric: a framework for assessing the quality
  of clinical notes* (2012). PDQI-9 dimensions adapted for LLM proxy evaluation.
- Radford A et al. *Robust Speech Recognition via Large-Scale Weak Supervision*.
  OpenAI Whisper (2022).
- Baayen-Schreuder H et al. *pyannote.audio 2.1 speaker diarization pipeline* (2023).

---

**Repository:** [ambient-scribe](https://github.com/PraveenKumarVk/ambient-scribe)
&nbsp;|&nbsp;
**Model:** claude-sonnet-4-6 · whisper large-v2 · pyannote/speaker-diarization-3.1
"""

# Shared output component definitions (reused across both pipeline tabs)
def _output_components() -> list:
    """
    Return fresh Gradio output components with render=False so they can be
    placed explicitly with .render().  Called twice (live tab + demo tab) to
    produce two independent sets of components.
    """
    transcript_df = gr.Dataframe(
        headers=["Speaker", "Start", "Text"],
        label="Transcript",
        wrap=True,
        interactive=False,
        render=False,
    )
    soap_s = gr.Textbox(label="Subjective",          lines=4, interactive=True,  render=False)
    soap_o = gr.Textbox(label="Objective",            lines=4, interactive=True,  render=False)
    soap_a = gr.Textbox(label="Assessment",           lines=4, interactive=True,  render=False)
    soap_p = gr.Textbox(label="Plan",                 lines=4, interactive=True,  render=False)
    hallucination_html = gr.HTML(value="", render=False)
    icd    = gr.Textbox(label="ICD-10 Suggestions",   lines=3, interactive=False, render=False)
    lat    = gr.Plot(label="Pipeline Latency",        render=False)
    wer    = gr.Dataframe(
        headers=["Category", "WER"],
        label="Medical ASR Accuracy",
        interactive=False,
        render=False,
    )
    hal    = gr.Dataframe(
        headers=["Grounded", "Claim", "Similarity", "Best Source"],
        label="Hallucination Flags",
        wrap=True,
        interactive=False,
        render=False,
    )
    pdqi   = gr.Plot(label="PDQI-9 Proxy Score", render=False)
    return [transcript_df, soap_s, soap_o, soap_a, soap_p, icd, lat, wer, hal, pdqi, hallucination_html]


def _render_outputs(components: list) -> None:
    """Place all 11 output components into the current layout context."""
    transcript_df, s, o, a, p, icd, lat, wer, hal, pdqi, hallucination_html = components
    transcript_df.render()
    with gr.Row():
        s.render()
        o.render()
    hallucination_html.render()
    with gr.Row():
        a.render()
        p.render()
    icd.render()
    with gr.Row():
        lat.render()
        pdqi.render()
    with gr.Row():
        wer.render()
        hal.render()


with gr.Blocks(
    title="Ambient Clinical Scribe",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=".gr-prose { max-width: 900px; margin: auto }",
) as demo:

    gr.Markdown(
        "# 🩺 Ambient Clinical Scribe\n"
        "AI-assisted clinical documentation · WhisperX · pyannote · Claude claude-sonnet-4-6\n\n"
        f"{_DISCLAIMER}"
    )

    with gr.Tabs():

        # ── Tab 1: Live Pipeline ────────────────────────────────────────────
        with gr.Tab("Live Pipeline"):
            gr.Markdown(_CPU_WARNING)

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Audio Input (mic or file)",
                    )
                    specialty_radio = gr.Radio(
                        choices=list(_SPECIALTY_MAP.keys()),
                        value="Primary Care",
                        label="Clinical Specialty",
                    )
                    submit_btn = gr.Button("▶  Transcribe & Generate Note", variant="primary")

                with gr.Column(scale=3):
                    live_comps = _output_components()
                    _render_outputs(live_comps)

            submit_btn.click(
                fn=run_pipeline,
                inputs=[audio_in, specialty_radio],
                outputs=live_comps,
                show_progress="full",
            )

        # ── Tab 2: Demo Encounters ──────────────────────────────────────────
        with gr.Tab("Demo Encounters"):
            gr.Markdown(
                "**⚠️ SYNTHETIC DATA ONLY — NOT FOR CLINICAL USE.**  \n"
                "These encounters were generated from scripted audio for demonstration "
                "purposes. All patient data is fictional."
            )

            _choices = _demo_choices()
            demo_dropdown = gr.Dropdown(
                choices=_choices,
                value=_choices[0][1] if _choices else None,
                label="Select Demo Encounter",
                info=(
                    "Run `backend/seed_demo_data.py` first to populate this list."
                    if not _choices else
                    f"{len(_choices)} demo encounter(s) available."
                ),
            )
            load_btn = gr.Button("Load Encounter", variant="primary")

            demo_comps = _output_components()
            _render_outputs(demo_comps)

            load_btn.click(
                fn=load_demo,
                inputs=[demo_dropdown],
                outputs=demo_comps,
            )

        # ── Tab 3: About ────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(_ABOUT_MD, elem_classes=["gr-prose"])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
