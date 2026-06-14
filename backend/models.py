from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# ASR Layer
# ---------------------------------------------------------------------------

class Word(BaseModel):
    start: float
    end: float
    word: str
    score: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list[Word] = []


class LatencyBreakdown(BaseModel):
    load_audio_ms: float = 0.0
    asr_ms: float = 0.0
    alignment_ms: float = 0.0
    diarization_ms: float = 0.0
    total_ms: float = 0.0


class TranscriptionResult(BaseModel):
    language: str
    segments: list[Segment]
    latency: LatencyBreakdown
    pipeline_used: str  # "whisper" | "lfm" | "hybrid"


# ---------------------------------------------------------------------------
# SOAP Layer
# ---------------------------------------------------------------------------

class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    chief_complaint: str
    icd10_suggestions: list[str] = []
    raw_claims: list[str] = []  # extracted A+P claims for hallucination eval


class SOAPResult(BaseModel):
    note: SOAPNote
    generation_ms: float
    specialty: str
    validation_warnings: list[str] = []


# ---------------------------------------------------------------------------
# Eval Layer
# ---------------------------------------------------------------------------

class MedicalWERResult(BaseModel):
    medications: float | None = None
    symptoms: float | None = None
    procedures: float | None = None
    overall_medical: float | None = None
    entity_counts: dict[str, int] = {}


class HallucinationFlag(BaseModel):
    claim: str
    grounded: bool
    max_similarity: float
    best_source_text: str
    best_source_speaker: str


class HallucinationResult(BaseModel):
    flags: list[HallucinationFlag]
    grounded_count: int
    ungrounded_count: int
    threshold_used: float = 0.45


class PDQIScores(BaseModel):
    accuracy: int
    completeness: int
    organization: int
    conciseness: int
    attribution: int


class PDQIResult(BaseModel):
    scores: PDQIScores
    reasoning: dict[str, str]
    mean_score: float
    disclaimer: str = (
        "LLM-proxy rubric adapted from PDQI-9 "
        "(Stetson et al., 2012). Not validated "
        "against human clinical raters."
    )


class SpeakerAttributionResult(BaseModel):
    accuracy: float
    correct: int
    total: int
    confusion: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Full Encounter
# ---------------------------------------------------------------------------

class EncounterMetrics(BaseModel):
    encounter_id: str
    latency: LatencyBreakdown
    medical_wer: MedicalWERResult | None = None
    hallucination: HallucinationResult | None = None
    pdqi: PDQIResult | None = None
    speaker_attribution: SpeakerAttributionResult | None = None
