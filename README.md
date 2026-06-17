---
title: Ambient Scribe
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
---

# Ambient Clinical Scribe

Physicians spend an estimated 2 hours on documentation for every 1 hour of direct patient care, a burden that is a leading driver of burnout and early exit from clinical practice. This pipeline converts a recorded or uploaded patient encounter into a structured SOAP note and scores that note against an automated eval suite, rather than leaving an unverified LLM output for the clinician to trust or re-check by hand.

## Live Demo

[**Try it on Hugging Face Spaces →**](https://huggingface.co/spaces/pvarkala18/ambient-scribe)

[![Loom demo thumbnail](https://cdn.loom.com/sessions/thumbnails/PLACEHOLDER-with-play.gif)](https://www.loom.com/share/PLACEHOLDER)

## Architecture

```
┌──────────────────┐   ┌─────┐   ┌─────────────┐   ┌─────────────┐
│ Microphone/Upload │──▶│ VAD │──▶│ WhisperX ASR│──▶│ Diarization │
└──────────────────┘   └─────┘   └─────────────┘   └──────┬──────┘
                                                            │
                                                            ▼
                        ┌────────────┐   ┌───────────┐   ┌────────────┐
                        │ Eval Suite │◀──│ SOAP Note │◀──│ Claude API │
                        └────────────┘   └───────────┘   └────────────┘
```

VAD trims silence before it reaches the ASR model. WhisperX produces word-level timestamps, which pyannote uses to assign each segment to SPEAKER_00 / SPEAKER_01. The diarized transcript goes to Claude to draft the SOAP note. The eval suite then scores that note independently of the model that generated it.

## Eval metrics

| Metric | Method | What it measures | Demo Result |
|---|---|---|---|
| Medical WER | Word Error Rate over `en_ner_bc5cdr_md` entity spans (medications, symptoms) between reference and hypothesis transcript | Whether ASR preserves clinically load-bearing terms, not just overall word accuracy | medications 8 %, symptoms 12 % (single seeded encounter) |
| Hallucination Rate | Each SOAP claim embedded with `all-MiniLM-L6-v2`, cosine-matched against transcript sentences, flagged ungrounded below 0.45 similarity | Fraction of generated claims with no traceable source in the transcript | 1 / 9 claims ungrounded (single seeded encounter) |
| PDQI-9 Proxy | Claude-as-judge scoring accuracy, completeness, organisation, conciseness, attribution (1–5 each) against the transcript | LLM-rubric approximation of physician-rated note quality; not a substitute for human review | mean 4.2 / 5 (single seeded encounter) |
| Latency | Wall-clock breakdown: ASR + alignment, diarization, SOAP generation, eval | End-to-end time budget per pipeline stage | ~38 s total (CPU, int8, large-v2) |
| Speaker Attribution | Diarization label compared against known speaker turns in synthetic transcripts | Whether physician and patient statements are attributed correctly in the downstream note | Qualitative pass; no held-out ground truth yet |

Numbers above come from a single seeded demo encounter, not a benchmark suite — see Limitations.

## Pipeline comparison

| | Whisper | LFM2.5-Audio | Hybrid |
|---|---|---|---|
| ASR | WhisperX (large-v2), word-level timestamp alignment | _Coming soon — architecture ready_ | _Coming soon — architecture ready_ |
| Diarization | Separate pyannote 3.1 pass over aligned words | _Coming soon — architecture ready_ | _Coming soon — architecture ready_ |
| Latency profile | Two model passes (ASR, then diarization) | _Coming soon — architecture ready_ | _Coming soon — architecture ready_ |
| Status | Implemented, default pipeline | _Coming soon — architecture ready_ | _Coming soon — architecture ready_ |

`route_pipeline()` in `backend/pipeline/asr.py` dispatches on a `pipeline` parameter; `LFMProvider` exists as a stub (`NotImplementedError`) so adding an end-to-end audio LLM does not require touching the API contract or the frontend.

## Limitations

- All audio in `demo_audio/` and `demo_data/` is synthetic or scripted — no real patient data was used to build or test this system.
- PDQI-9 proxy scores come from an LLM judge (Claude), not licensed clinicians; treat them as relative signal, not validated quality measurement.
- Medical WER for procedures is always `null` — `en_ner_bc5cdr_md` has no procedure entity type, so that category is currently unmeasured.
- Hallucination detection uses sentence-level cosine similarity, which can miss claims that are true but phrased differently from the transcript, and can accept claims that are false but lexically close.
- Speaker attribution assumes exactly two speakers (`num_speakers=2`); it has not been tested on multi-party encounters.
- No authentication on any endpoint — this is a local or demo deployment and must not be exposed to real patient data without appropriate access controls and a HIPAA-compliant infrastructure review.

## Tech stack

| Layer | Technology |
|---|---|
| ASR | WhisperX (large-v2), int8 quantisation on CPU |
| Diarization | pyannote/speaker-diarization-3.1 |
| Note generation | Claude (claude-sonnet-4-6) via Anthropic API |
| Eval — NER | scispaCy `en_ner_bc5cdr_md` |
| Eval — semantic similarity | sentence-transformers `all-MiniLM-L6-v2` |
| Eval — WER | jiwer |
| Data layer | SQLAlchemy 2.x, SQLite |
| Frontend | Gradio 5.x, Plotly |
| Deployment | Hugging Face Spaces (Gradio SDK) |

## Local setup

```bash
git clone https://huggingface.co/spaces/pvarkala18/ambient-scribe
cd ambient-scribe
pip install -r requirements.txt          # includes scispaCy model wheel
export ANTHROPIC_API_KEY=sk-ant-...      # required for SOAP generation
export HF_TOKEN=hf_...                   # required for pyannote diarization
export WHISPER_MODEL_SIZE=tiny           # use large-v2 for production quality
python app.py                            # starts on http://localhost:7860
```

`DEVICE` defaults to `cpu` and `COMPUTE_TYPE` to `int8`. Override both if you have a GPU. The eval suite is optional — the pipeline degrades gracefully if `ANTHROPIC_API_KEY` is absent for the PDQI judge, but SOAP generation will fail without it.

---

> **Synthetic data notice**
>
> **All encounters in this repository — audio files, transcripts, and the seeded SQLite database — are fully synthetic.** They were generated programmatically for demonstration purposes and contain no real patient information. This system has not been evaluated on clinical data and is not intended for use in any clinical workflow.

---

## Citation

PDQI-9 instrument:

> Stetson, P. D., Bakken, S., Wrenn, J. O., & Siegler, E. L. (2012). Assessing physician use of chart review summaries: A randomized trial. *Journal of the American Medical Informatics Association*, 19(4), 164–174. https://doi.org/10.1136/amiajnl-2011-000533
