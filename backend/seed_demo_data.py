#!/usr/bin/env python3
"""
Seed demo encounters into ambient_scribe.db and export to demo_data/demo.db.

Usage (from project root "Ambient Scribe/"):
    PYTHONPATH=. .venv/bin/python backend/seed_demo_data.py

Expects audio files:
    demo_audio/encounter-a.wav
    demo_audio/encounter-b.wav
    demo_audio/encounter-c.wav

Optional ground-truth files (enables WER + speaker attribution):
    demo_data/encounter_a.json
    demo_data/encounter_b.json
    demo_data/encounter_c.json

Ground-truth JSON schema:
{
    "reference_transcript": "SPEAKER_00: Hello...\nSPEAKER_01: I've been...",
    "speaker_segments": [
        {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "Hello..."},
        {"start": 3.5, "end": 8.1, "speaker": "SPEAKER_01", "text": "I've been..."}
    ]
}
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402 — triggers dotenv load
from backend.db import (  # noqa: E402
    _DB_PATH,
    _engine,
    create_encounter,
    save_metrics,
    update_encounter_results,
    update_encounter_status,
)
from backend.models import Segment, SpeakerAttributionResult  # noqa: E402
from backend.pipeline.asr import WhisperXProvider  # noqa: E402
from backend.pipeline.soap import generate_soap, validate_soap  # noqa: E402

try:
    from backend.pipeline.eval import run_full_eval as _run_full_eval
    _EVAL_AVAILABLE = True
except (ImportError, AttributeError) as _eval_err:
    _EVAL_AVAILABLE = False
    print(f"[WARN] eval pipeline unavailable ({_eval_err}) — skipping WER / hallucination / PDQI")

# ---------------------------------------------------------------------------
# Demo encounter definitions
# ---------------------------------------------------------------------------

DEMO_AUDIO_DIR = ROOT / "demo_audio"
DEMO_DATA_DIR  = ROOT / "demo_data"
DEMO_DB_PATH   = DEMO_DATA_DIR / "demo.db"


class DemoConfig(NamedTuple):
    encounter_id:      str
    audio_path:        Path
    ground_truth_path: Path
    specialty:         str
    label:             str


DEMO_CONFIGS: list[DemoConfig] = [
    DemoConfig(
        encounter_id="demo-encounter-a",
        audio_path=DEMO_AUDIO_DIR / "encounter-a.wav",
        ground_truth_path=DEMO_DATA_DIR / "encounter_a.json",
        specialty="primary_care",
        label="Encounter A",
    ),
    DemoConfig(
        encounter_id="demo-encounter-b",
        audio_path=DEMO_AUDIO_DIR / "encounter-b.wav",
        ground_truth_path=DEMO_DATA_DIR / "encounter_b.json",
        specialty="primary_care",
        label="Encounter B",
    ),
    DemoConfig(
        encounter_id="demo-encounter-c",
        audio_path=DEMO_AUDIO_DIR / "encounter-c.wav",
        ground_truth_path=DEMO_DATA_DIR / "encounter_c.json",
        specialty="cardiology",
        label="Encounter C",
    ),
]

# ---------------------------------------------------------------------------
# Speaker attribution
# ---------------------------------------------------------------------------

def _compute_speaker_attribution(
    predicted_segments: list[Segment],
    gt_segments: list[dict],
) -> SpeakerAttributionResult:
    """
    Match predicted speaker labels to ground truth using maximum segment overlap.
    Tries both SPEAKER_00/SPEAKER_01 orientations to handle pyannote speaker-flip.
    """
    if not gt_segments or not predicted_segments:
        return SpeakerAttributionResult(accuracy=0.0, correct=0, total=0)

    gt = [
        (float(s["start"]), float(s["end"]), s["speaker"])
        for s in gt_segments
    ]

    def _best_gt_speaker(seg: Segment) -> str | None:
        best_overlap, best_spk = 0.0, None
        for gs, ge, gsp in gt:
            overlap = max(0.0, min(seg.end, ge) - max(seg.start, gs))
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, gsp
        return best_spk

    pairs: list[tuple[str, str]] = []
    for seg in predicted_segments:
        gt_spk = _best_gt_speaker(seg)
        if gt_spk is not None:
            pairs.append((seg.speaker, gt_spk))

    if not pairs:
        return SpeakerAttributionResult(
            accuracy=0.0, correct=0, total=len(predicted_segments)
        )

    def _score(mapping: dict[str, str]) -> int:
        return sum(1 for pred, gt_s in pairs if mapping.get(pred) == gt_s)

    score_canonical = _score({"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"})
    score_flipped   = _score({"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
    best_score = max(score_canonical, score_flipped)

    return SpeakerAttributionResult(
        accuracy=round(best_score / len(pairs), 4),
        correct=best_score,
        total=len(pairs),
    )

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _delete_demo_rows(encounter_ids: list[str]) -> None:
    """Remove any existing rows for these IDs so the script is idempotent."""
    from sqlalchemy import delete
    from sqlalchemy.orm import Session as DBSession
    from backend.db import (
        Encounter,
        EncounterEdit,
        EncounterMetrics as _DBMetrics,
    )

    with DBSession(_engine) as session:
        session.execute(
            delete(EncounterEdit).where(EncounterEdit.encounter_id.in_(encounter_ids))
        )
        session.execute(
            delete(_DBMetrics).where(_DBMetrics.encounter_id.in_(encounter_ids))
        )
        session.execute(
            delete(Encounter).where(Encounter.id.in_(encounter_ids))
        )
        session.commit()


def _update_speaker_accuracy(encounter_id: str, accuracy: float) -> None:
    from sqlalchemy import update
    from sqlalchemy.orm import Session as DBSession
    from backend.db import EncounterMetrics as _DBMetrics

    with DBSession(_engine) as session:
        session.execute(
            update(_DBMetrics)
            .where(_DBMetrics.encounter_id == encounter_id)
            .values(speaker_attribution_accuracy=accuracy)
        )
        session.commit()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_demo_db(seeded_ids: list[str]) -> None:
    """Copy main DB to demo_data/demo.db, keeping only seeded demo rows."""
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(_DB_PATH), str(DEMO_DB_PATH))

    placeholders = ",".join("?" * len(seeded_ids))
    con = sqlite3.connect(str(DEMO_DB_PATH))
    try:
        con.execute(
            f"DELETE FROM encounter WHERE id NOT IN ({placeholders})", seeded_ids
        )
        con.execute(
            f"DELETE FROM encounter_metrics WHERE encounter_id NOT IN ({placeholders})",
            seeded_ids,
        )
        con.execute(
            f"DELETE FROM encounter_edit WHERE encounter_id NOT IN ({placeholders})",
            seeded_ids,
        )
        con.commit()
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()

    size_kb = DEMO_DB_PATH.stat().st_size // 1024
    print(f"\n[OK] Exported {DEMO_DB_PATH.name}  ({size_kb} KB)  →  {DEMO_DB_PATH}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(value: object, fmt: str, suffix: str = "", fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        return format(value, fmt) + suffix
    except (TypeError, ValueError):
        return fallback


def _print_summary(rows: list[dict]) -> None:
    headers = ["Encounter", "Duration", "Total Latency", "Med WER", "Hallucinations", "PDQI Mean"]
    widths  = [15,          10,          16,              12,         16,               10]

    def _line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    sep = "─" * (sum(widths) + 2 * (len(widths) - 1))
    print(f"\n{sep}")
    print(_line(headers))
    print(sep)
    for r in rows:
        wer_val = r.get("wer")
        print(_line([
            r["label"],
            _fmt(r.get("duration"), ".1f", "s"),
            _fmt(r.get("total_ms"), ",.0f", " ms"),
            _fmt(wer_val * 100 if wer_val is not None else None, ".1f", "%", "N/A"),
            str(r["hallucinations"]) if r.get("hallucinations") is not None else "N/A",
            _fmt(r.get("pdqi"), ".2f", fallback="N/A"),
        ]))
    print(sep)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed() -> None:
    config.require_anthropic_api_key()

    available = [cfg for cfg in DEMO_CONFIGS if cfg.audio_path.exists()]
    skipped   = [cfg for cfg in DEMO_CONFIGS if not cfg.audio_path.exists()]

    if not available:
        print(f"[ERROR] No demo audio files found in {DEMO_AUDIO_DIR}")
        print("  Expected: encounter-a.wav  encounter-b.wav  encounter-c.wav")
        sys.exit(1)

    if skipped:
        print(f"[WARN] Skipping {len(skipped)} missing audio file(s):")
        for cfg in skipped:
            print(f"       {cfg.audio_path}")

    print(f"\nSeeding {len(available)} demo encounter(s) …")
    _delete_demo_rows([cfg.encounter_id for cfg in DEMO_CONFIGS])

    provider = WhisperXProvider()
    summary_rows: list[dict] = []
    seeded_ids:   list[str]  = []

    for cfg in available:
        bar = "─" * 62
        print(f"\n{bar}")
        print(f"  {cfg.label}  ({cfg.audio_path.name})  specialty={cfg.specialty}")
        print(bar)

        row: dict = dict(
            label=cfg.label, duration=None, total_ms=None,
            wer=None, hallucinations=None, pdqi=None,
        )

        try:
            # ── Ground truth ──────────────────────────────────────────────
            gt: dict = {}
            if cfg.ground_truth_path.exists():
                with open(cfg.ground_truth_path) as fh:
                    gt = json.load(fh)
                print(f"  [GT ] Loaded {cfg.ground_truth_path.name}")
            else:
                print(f"  [GT ] No ground truth ({cfg.ground_truth_path.name}) "
                      "— WER + attribution skipped")

            # ── Create DB row ─────────────────────────────────────────────
            create_encounter(
                id=cfg.encounter_id,
                audio_filename=cfg.audio_path.name,
                specialty=cfg.specialty,
                pipeline="whisperx",
                is_demo=True,
            )

            # ── Transcription ─────────────────────────────────────────────
            print("  [1/4] Transcription … ", end="", flush=True)
            tr = provider.transcribe(str(cfg.audio_path))
            duration = max((s.end for s in tr.segments), default=0.0)
            row["duration"] = duration
            print(
                f"done  {tr.latency.total_ms:,.0f} ms  "
                f"{len(tr.segments)} segments  {duration:.1f}s audio"
            )

            raw_transcript = " ".join(s.text.strip() for s in tr.segments)
            diarized = [s.model_dump() for s in tr.segments]

            # ── SOAP generation ───────────────────────────────────────────
            print("  [2/4] SOAP generation … ", end="", flush=True)
            soap = generate_soap(tr.segments, specialty=cfg.specialty)
            warnings_list = validate_soap(soap)
            if warnings_list:
                print(f"done  {soap.generation_ms:,.0f} ms  "
                      f"[{len(warnings_list)} warning(s)]")
            else:
                print(f"done  {soap.generation_ms:,.0f} ms")

            soap_dict = soap.model_dump()
            soap_dict["validation_warnings"] = warnings_list
            update_encounter_results(
                cfg.encounter_id,
                raw_transcript=raw_transcript,
                diarized_segments=diarized,
                soap_note=soap_dict,
            )

            total_ms = tr.latency.total_ms + soap.generation_ms
            row["total_ms"] = total_ms

            # ── Evaluation ────────────────────────────────────────────────
            ref_transcript = gt.get("reference_transcript")

            if _EVAL_AVAILABLE:
                print("  [3/4] Evaluation … ", end="", flush=True)
                metrics_obj = _run_full_eval(
                    encounter_id=cfg.encounter_id,
                    reference_transcript=ref_transcript,
                    hypothesis_transcript=raw_transcript,
                    soap_result=soap,
                    segments=tr.segments,
                    latency=tr.latency,
                )
                print("done")

                if metrics_obj.medical_wer:
                    wers = [
                        v for v in (
                            metrics_obj.medical_wer.medications,
                            metrics_obj.medical_wer.symptoms,
                        )
                        if v is not None
                    ]
                    row["wer"] = round(sum(wers) / len(wers), 4) if wers else None

                if metrics_obj.hallucination:
                    row["hallucinations"] = metrics_obj.hallucination.ungrounded_count

                if metrics_obj.pdqi:
                    row["pdqi"] = metrics_obj.pdqi.mean_score

            else:
                print("  [3/4] Evaluation … skipped")
                save_metrics(cfg.encounter_id, {
                    "vad_ms":         tr.latency.load_audio_ms,
                    "asr_ms":         tr.latency.asr_ms + tr.latency.alignment_ms,
                    "diarization_ms": tr.latency.diarization_ms,
                    "note_gen_ms":    soap.generation_ms,
                    "total_ms":       total_ms,
                })

            # ── Speaker attribution ───────────────────────────────────────
            gt_segments = gt.get("speaker_segments", [])
            if gt_segments:
                print("  [4/4] Speaker attribution … ", end="", flush=True)
                attr = _compute_speaker_attribution(tr.segments, gt_segments)
                _update_speaker_accuracy(cfg.encounter_id, attr.accuracy)
                print(
                    f"done  {attr.correct}/{attr.total} correct  "
                    f"({attr.accuracy*100:.1f}%)"
                )
            else:
                print("  [4/4] Speaker attribution … skipped (no GT segments)")

            update_encounter_status(cfg.encounter_id, "complete")
            seeded_ids.append(cfg.encounter_id)
            print(f"  ✓  {cfg.label} complete")

        except Exception as exc:
            print(f"\n  [ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            try:
                update_encounter_status(
                    cfg.encounter_id,
                    "failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass  # DB row may not exist if create_encounter failed

        summary_rows.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        _print_summary(summary_rows)

    # ── Export demo.db ────────────────────────────────────────────────────────
    if seeded_ids:
        _export_demo_db(seeded_ids)
        print(f"\n  Seeded {len(seeded_ids)}/{len(available)} encounter(s) successfully.")
        print("  Commit demo_data/demo.db to the repo so HF Spaces loads it on startup.")
    else:
        print("\n[WARN] No encounters seeded successfully — demo_data/demo.db not updated.")
        sys.exit(1)


if __name__ == "__main__":
    seed()
