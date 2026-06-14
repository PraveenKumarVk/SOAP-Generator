from __future__ import annotations

import difflib
import json
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from backend import db
from backend.db import (
    EncounterMetrics as DBMetricsRow,
    _engine as _db_engine,
    create_encounter,
    get_demo_encounters,
    get_encounter,
    list_encounters,
    save_edit,
    save_metrics,
    update_encounter_results,
    update_encounter_status,
)
from backend.pipeline.asr import WhisperXProvider
from backend.pipeline.soap import generate_soap, validate_soap

try:
    from backend.pipeline.eval import run_full_eval as _run_full_eval  # type: ignore[import]
    _EVAL_AVAILABLE = True
except (ImportError, AttributeError):
    _EVAL_AVAILABLE = False

_UPLOADS_DIR = Path(__file__).resolve().parent.parent / "audio_uploads"

_provider_lock = threading.Lock()
_whisperx_provider: WhisperXProvider | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Ambient Scribe API", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_provider() -> WhisperXProvider:
    global _whisperx_provider
    if _whisperx_provider is None:
        with _provider_lock:
            if _whisperx_provider is None:
                _whisperx_provider = WhisperXProvider()
    return _whisperx_provider


def _fetch_metrics(encounter_id: str) -> dict | None:
    with DBSession(_db_engine) as session:
        row = session.scalars(
            select(DBMetricsRow).where(DBMetricsRow.encounter_id == encounter_id)
        ).first()
        if row is None:
            return None
        return {
            "vad_ms": row.vad_ms,
            "asr_ms": row.asr_ms,
            "diarization_ms": row.diarization_ms,
            "note_gen_ms": row.note_gen_ms,
            "total_ms": row.total_ms,
            "medical_wer_medications": row.medical_wer_medications,
            "medical_wer_symptoms": row.medical_wer_symptoms,
            "medical_wer_procedures": row.medical_wer_procedures,
            "hallucination_flags": json.loads(row.hallucination_flags)
                if row.hallucination_flags else None,
            "hallucination_count": row.hallucination_count,
            "pdqi_scores": json.loads(row.pdqi_scores) if row.pdqi_scores else None,
            "pdqi_mean": row.pdqi_mean,
            "speaker_attribution_accuracy": row.speaker_attribution_accuracy,
        }


# ---------------------------------------------------------------------------
# Background pipeline task
# ---------------------------------------------------------------------------

def run_full_pipeline(encounter_id: str, audio_path: str, specialty: str) -> None:
    try:
        update_encounter_status(encounter_id, "processing")

        provider = _get_provider()
        tr = provider.transcribe(audio_path)

        raw_transcript = " ".join(s.text.strip() for s in tr.segments)
        diarized = [s.model_dump() for s in tr.segments]

        soap = generate_soap(tr.segments, specialty=specialty)
        warnings = validate_soap(soap)

        soap_dict = soap.model_dump()
        soap_dict["validation_warnings"] = warnings

        update_encounter_results(
            encounter_id,
            raw_transcript=raw_transcript,
            diarized_segments=diarized,
            soap_note=soap_dict,
        )

        if _EVAL_AVAILABLE:
            _run_full_eval(
                encounter_id=encounter_id,
                reference_transcript=None,  # ground truth only available for demo encounters
                hypothesis_transcript=raw_transcript,
                soap_result=soap,
                segments=tr.segments,
                latency=tr.latency,
            )
        else:
            save_metrics(encounter_id, {
                "vad_ms": tr.latency.load_audio_ms,
                "asr_ms": tr.latency.asr_ms + tr.latency.alignment_ms,
                "diarization_ms": tr.latency.diarization_ms,
                "note_gen_ms": soap.generation_ms,
                "total_ms": tr.latency.total_ms + soap.generation_ms,
            })

        update_encounter_status(encounter_id, "complete")

    except Exception as exc:
        update_encounter_status(
            encounter_id,
            "failed",
            error_message=f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    try:
        list_encounters(limit=1)
        db_status = "ok"
    except Exception:
        db_status = "error"

    return {
        "status": "ok",
        "db": db_status,
        "whisperx": "loaded" if _whisperx_provider is not None else "not_loaded",
    }


@app.post("/encounter/upload")
async def upload_encounter(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    specialty: str = Form(default="primary_care"),
) -> dict:
    encounter_id = str(uuid.uuid4())
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    save_path = _UPLOADS_DIR / f"{encounter_id}{suffix}"

    async with aiofiles.open(save_path, "wb") as fh:
        while chunk := await audio_file.read(1024 * 1024):  # stream in 1 MB chunks
            await fh.write(chunk)

    create_encounter(
        id=encounter_id,
        audio_filename=audio_file.filename or save_path.name,
        specialty=specialty,
        pipeline="whisperx",
    )

    background_tasks.add_task(run_full_pipeline, encounter_id, str(save_path), specialty)

    return {"encounter_id": encounter_id, "status": "processing"}


@app.get("/encounters/demo")
def demo_encounters_route() -> list[dict]:
    return get_demo_encounters()


@app.get("/encounters")
def list_encounters_route() -> list[dict]:
    rows = list_encounters(limit=20)
    for row in rows:
        row.pop("raw_transcript", None)
        row.pop("diarized_segments", None)
    return rows


@app.get("/encounter/{encounter_id}")
def get_encounter_route(encounter_id: str) -> dict:
    try:
        enc = get_encounter(encounter_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Encounter {encounter_id!r} not found")

    enc["metrics"] = _fetch_metrics(encounter_id)
    return enc


class _EditBody(BaseModel):
    section: str
    edited_text: str


_SECTION_TO_FIELD = {"S": "subjective", "O": "objective", "A": "assessment", "P": "plan"}


@app.post("/encounter/{encounter_id}/edit")
def edit_section(encounter_id: str, body: _EditBody) -> dict:
    try:
        enc = get_encounter(encounter_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Encounter {encounter_id!r} not found")

    soap = enc.get("soap_note") or {}
    # soap_note is stored as {"note": {...}, ...} (SOAPResult.model_dump layout)
    note_fields = soap.get("note", soap)

    field_name = _SECTION_TO_FIELD.get(body.section, body.section)
    original_text: str = note_fields.get(field_name, "")

    diff_lines = list(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            body.edited_text.splitlines(keepends=True),
            fromfile=f"{body.section} (original)",
            tofile=f"{body.section} (edited)",
        )
    )

    save_edit(
        encounter_id=encounter_id,
        section=body.section,
        original=original_text,
        edited=body.edited_text,
        diff_unified="".join(diff_lines),
    )

    return {"saved": True, "diff_lines": len(diff_lines)}
