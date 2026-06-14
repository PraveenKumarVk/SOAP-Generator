import json
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print(f"Re-running with project venv: {VENV_PYTHON}", flush=True)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import EncounterMetrics as DBMetricsRow
from backend.db import _engine as db_engine
from backend.models import LatencyBreakdown, Segment, SOAPNote, SOAPResult
from backend.pipeline.eval import (
    check_hallucinations,
    compute_medical_wer,
    compute_pdqi_proxy,
    run_full_eval,
)


REFERENCE_TRANSCRIPT = (
    "Patient reports headaches and hypertension. "
    "Patient takes atorvastatin forty milligrams once daily."
)
HYPOTHESIS_TRANSCRIPT = (
    "Patient reports headaches and hypertension. "
    "Patient takes atorvastatin 40 milligrams daily."
)

SEGMENTS = [
    Segment(
        start=0.0,
        end=5.0,
        speaker="SPEAKER_01",
        text="I have headaches and hypertension.",
    ),
    Segment(
        start=5.0,
        end=10.0,
        speaker="SPEAKER_01",
        text="I take atorvastatin forty milligrams once daily.",
    ),
    Segment(
        start=10.0,
        end=15.0,
        speaker="SPEAKER_00",
        text="We will monitor your blood pressure and follow up in four weeks.",
    ),
]

GOOD_SOAP = SOAPResult(
    note=SOAPNote(
        subjective=(
            "Patient reports headaches, hypertension, and current atorvastatin use."
        ),
        objective="No exam findings or vitals were provided in this sample transcript.",
        assessment="Hypertension with headaches.",
        plan="Continue atorvastatin and monitor blood pressure with follow-up.",
        chief_complaint="Headaches with hypertension.",
        icd10_suggestions=[
            "I10: Essential (primary) hypertension",
            "R51.9: Headache, unspecified",
        ],
        raw_claims=[
            "Patient has hypertension.",
            "Patient reports headaches.",
            "Patient takes atorvastatin.",
            "Blood pressure monitoring is planned.",
        ],
    ),
    generation_ms=123.0,
    specialty="primary_care",
)

BAD_SOAP = SOAPResult(
    note=SOAPNote(
        subjective=GOOD_SOAP.note.subjective,
        objective=GOOD_SOAP.note.objective,
        assessment="Hypertension with headaches and pneumonia.",
        plan="Continue atorvastatin and start antibiotics for pneumonia.",
        chief_complaint=GOOD_SOAP.note.chief_complaint,
        icd10_suggestions=[
            "I10: Essential (primary) hypertension",
            "J18.9: Pneumonia, unspecified organism",
        ],
        raw_claims=[
            "Patient has hypertension.",
            "Patient takes atorvastatin.",
            "Patient has pneumonia.",
            "Antibiotics are being started for pneumonia.",
        ],
    ),
    generation_ms=123.0,
    specialty="primary_care",
)


medical_wer = compute_medical_wer(REFERENCE_TRANSCRIPT, HYPOTHESIS_TRANSCRIPT)
print("Medical WER:", medical_wer.model_dump())

assert medical_wer.medications is not None
assert medical_wer.symptoms is not None
assert medical_wer.entity_counts["medications"] > 0
assert medical_wer.entity_counts["symptoms"] > 0

hallucination = check_hallucinations(BAD_SOAP, SEGMENTS)
print("Hallucination flags:", [flag.model_dump() for flag in hallucination.flags])

ungrounded_claims = [flag.claim for flag in hallucination.flags if not flag.grounded]
assert any("pneumonia" in claim.lower() for claim in ungrounded_claims)
assert hallucination.ungrounded_count > 0

pdqi = compute_pdqi_proxy(HYPOTHESIS_TRANSCRIPT, GOOD_SOAP.model_dump())
print("PDQI:", pdqi.model_dump())

pdqi_scores = pdqi.scores.model_dump()
assert pdqi_scores
assert all(1 <= score <= 5 for score in pdqi_scores.values())

encounter_id = f"test-eval-{uuid.uuid4()}"
metrics = run_full_eval(
    encounter_id=encounter_id,
    reference_transcript=REFERENCE_TRANSCRIPT,
    hypothesis_transcript=HYPOTHESIS_TRANSCRIPT,
    soap_result=GOOD_SOAP,
    segments=SEGMENTS,
    latency=LatencyBreakdown(
        load_audio_ms=10.0,
        asr_ms=20.0,
        alignment_ms=30.0,
        diarization_ms=40.0,
        total_ms=100.0,
    ),
)
print("Full eval:", metrics.model_dump())

with Session(db_engine) as session:
    row = session.scalars(
        select(DBMetricsRow).where(DBMetricsRow.encounter_id == encounter_id)
    ).first()

assert row is not None
assert row.medical_wer_medications is not None
assert row.medical_wer_symptoms is not None
assert row.hallucination_flags is not None
assert row.hallucination_count is not None
assert row.pdqi_scores is not None
assert row.pdqi_mean is not None

saved_flags = json.loads(row.hallucination_flags)
saved_pdqi = json.loads(row.pdqi_scores)
print("Saved hallucination flags:", saved_flags)
print("Saved PDQI scores:", saved_pdqi)
