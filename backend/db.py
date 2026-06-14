from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

_DB_PATH = Path(__file__).resolve().parent.parent / "ambient_scribe.db"
_engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})


class _Base(DeclarativeBase):
    pass


class Encounter(_Base):
    __tablename__ = "encounter"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    audio_filename: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    specialty: Mapped[str] = mapped_column(String, default="primary_care")
    pipeline: Mapped[str] = mapped_column(String, default="whisper")
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    diarized_segments: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    soap_note: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    metrics: Mapped[list[EncounterMetrics]] = relationship(
        "EncounterMetrics", back_populates="encounter", cascade="all, delete-orphan"
    )
    edits: Mapped[list[EncounterEdit]] = relationship(
        "EncounterEdit", back_populates="encounter", cascade="all, delete-orphan"
    )


class EncounterMetrics(_Base):
    __tablename__ = "encounter_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    encounter_id: Mapped[str] = mapped_column(String, ForeignKey("encounter.id"), nullable=False)
    vad_ms: Mapped[float] = mapped_column(Float, nullable=False)
    asr_ms: Mapped[float] = mapped_column(Float, nullable=False)
    diarization_ms: Mapped[float] = mapped_column(Float, nullable=False)
    note_gen_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_ms: Mapped[float] = mapped_column(Float, nullable=False)
    medical_wer_medications: Mapped[float | None] = mapped_column(Float, nullable=True)
    medical_wer_symptoms: Mapped[float | None] = mapped_column(Float, nullable=True)
    medical_wer_procedures: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_flags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    hallucination_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pdqi_scores: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    pdqi_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    speaker_attribution_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    encounter: Mapped[Encounter] = relationship("Encounter", back_populates="metrics")


class EncounterEdit(_Base):
    __tablename__ = "encounter_edit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    encounter_id: Mapped[str] = mapped_column(String, ForeignKey("encounter.id"), nullable=False)
    edited_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    section: Mapped[str] = mapped_column(String, nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    edited_text: Mapped[str] = mapped_column(Text, nullable=False)
    diff_unified: Mapped[str] = mapped_column(Text, nullable=False)

    encounter: Mapped[Encounter] = relationship("Encounter", back_populates="edits")


def create_tables() -> None:
    _Base.metadata.create_all(_engine)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_encounter(
    id: str,
    audio_filename: str,
    specialty: str = "primary_care",
    pipeline: str = "whisper",
    is_demo: bool = False,
) -> None:
    with Session(_engine) as session:
        session.add(
            Encounter(
                id=id,
                audio_filename=audio_filename,
                specialty=specialty,
                pipeline=pipeline,
                status="processing",
                is_demo=is_demo,
            )
        )
        session.commit()


def update_encounter_status(
    id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    with Session(_engine) as session:
        enc = session.get(Encounter, id)
        if enc is None:
            raise KeyError(f"Encounter {id!r} not found")
        enc.status = status
        if error_message is not None:
            enc.error_message = error_message
        session.commit()


def update_encounter_results(
    id: str,
    raw_transcript: str,
    diarized_segments: list | dict,
    soap_note: dict,
) -> None:
    with Session(_engine) as session:
        enc = session.get(Encounter, id)
        if enc is None:
            raise KeyError(f"Encounter {id!r} not found")
        enc.raw_transcript = raw_transcript
        enc.diarized_segments = json.dumps(diarized_segments)
        enc.soap_note = json.dumps(soap_note)
        enc.status = "complete"
        session.commit()


def save_metrics(encounter_id: str, metrics_dict: dict) -> None:
    with Session(_engine) as session:
        row = EncounterMetrics(
            encounter_id=encounter_id,
            vad_ms=metrics_dict.get("vad_ms", 0.0),
            asr_ms=metrics_dict.get("asr_ms", 0.0),
            diarization_ms=metrics_dict.get("diarization_ms", 0.0),
            note_gen_ms=metrics_dict.get("note_gen_ms", 0.0),
            total_ms=metrics_dict.get("total_ms", 0.0),
            medical_wer_medications=metrics_dict.get("medical_wer_medications"),
            medical_wer_symptoms=metrics_dict.get("medical_wer_symptoms"),
            medical_wer_procedures=metrics_dict.get("medical_wer_procedures"),
            hallucination_flags=_json_or_none(metrics_dict.get("hallucination_flags")),
            hallucination_count=metrics_dict.get("hallucination_count"),
            pdqi_scores=_json_or_none(metrics_dict.get("pdqi_scores")),
            pdqi_mean=metrics_dict.get("pdqi_mean"),
            speaker_attribution_accuracy=metrics_dict.get("speaker_attribution_accuracy"),
        )
        session.add(row)
        session.commit()


def save_edit(
    encounter_id: str,
    section: str,
    original: str,
    edited: str,
    diff_unified: str = "",
) -> None:
    with Session(_engine) as session:
        session.add(
            EncounterEdit(
                encounter_id=encounter_id,
                section=section,
                original_text=original,
                edited_text=edited,
                diff_unified=diff_unified,
            )
        )
        session.commit()


def get_encounter(id: str) -> dict:
    with Session(_engine) as session:
        enc = session.get(Encounter, id)
        if enc is None:
            raise KeyError(f"Encounter {id!r} not found")
        return _encounter_to_dict(enc)


def list_encounters(limit: int = 20) -> list[dict]:
    from sqlalchemy import select

    with Session(_engine) as session:
        rows = session.scalars(
            select(Encounter).order_by(Encounter.created_at.desc()).limit(limit)
        ).all()
        return [_encounter_to_dict(r) for r in rows]


def get_demo_encounters() -> list[dict]:
    from sqlalchemy import select

    with Session(_engine) as session:
        rows = session.scalars(
            select(Encounter)
            .where(Encounter.is_demo == True)  # noqa: E712
            .order_by(Encounter.created_at.desc())
        ).all()
        return [_encounter_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encounter_to_dict(enc: Encounter) -> dict:
    return {
        "id": enc.id,
        "created_at": enc.created_at.isoformat() if enc.created_at else None,
        "audio_filename": enc.audio_filename,
        "duration_seconds": enc.duration_seconds,
        "specialty": enc.specialty,
        "pipeline": enc.pipeline,
        "status": enc.status,
        "error_message": enc.error_message,
        "raw_transcript": enc.raw_transcript,
        "diarized_segments": json.loads(enc.diarized_segments) if enc.diarized_segments else None,
        "soap_note": json.loads(enc.soap_note) if enc.soap_note else None,
        "is_demo": enc.is_demo,
    }


def _json_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


create_tables()
