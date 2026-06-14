import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print(f"Re-running with project venv: {VENV_PYTHON}", flush=True)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline.asr import Segment
from backend.pipeline.soap import generate_soap, validate_soap


def _segment(index: int, speaker: str, text: str) -> Segment:
    return Segment(
        start=float(index * 5),
        end=float((index + 1) * 5),
        speaker=speaker,
        text=text,
    )


segments = [
    _segment(0, "SPEAKER_00", "Good morning. What brings you in today?"),
    _segment(
        1,
        "SPEAKER_01",
        "Hi doctor. I've been getting headaches for the last couple of weeks and I wanted to get them checked out.",
    ),
    _segment(
        2,
        "SPEAKER_00",
        "Can you describe the headaches? Where are they located and how often are they occurring?",
    ),
    _segment(
        3,
        "SPEAKER_01",
        "They usually start around my forehead and sometimes behind my eyes. I've been getting them about four or five times a week.",
    ),
    _segment(4, "SPEAKER_00", "On a scale from one to ten, how severe is the pain?"),
    _segment(
        5,
        "SPEAKER_01",
        "Usually around a six. Sometimes it gets up to an eight if I'm working on the computer for a long time.",
    ),
    _segment(
        6,
        "SPEAKER_00",
        "Have you noticed any vision changes, nausea, dizziness, or weakness?",
    ),
    _segment(
        7,
        "SPEAKER_01",
        "No weakness. I do occasionally feel a little dizzy, but I haven't had any vision problems.",
    ),
    _segment(8, "SPEAKER_00", "Are you currently taking any medications?"),
    _segment(
        9,
        "SPEAKER_01",
        "Yes. I'm taking atorvastatin forty milligrams once daily for cholesterol. I've been on it for about a year.",
    ),
    _segment(10, "SPEAKER_00", "Any other medications or supplements?"),
    _segment(11, "SPEAKER_01", "Just a daily multivitamin. Nothing else."),
    _segment(12, "SPEAKER_00", "Have you been checking your blood pressure recently?"),
    _segment(
        13,
        "SPEAKER_01",
        "Actually yes. I checked it at a pharmacy last week and it was around one hundred forty-five over ninety.",
    ),
    _segment(
        14,
        "SPEAKER_00",
        "That's higher than we would like. Do you have a history of hypertension?",
    ),
    _segment(
        15,
        "SPEAKER_01",
        "My primary care doctor mentioned elevated blood pressure last year, but I wasn't started on any medication.",
    ),
    _segment(16, "SPEAKER_00", "How has your sleep been lately?"),
    _segment(
        17,
        "SPEAKER_01",
        "Not great. I've been sleeping about five or six hours a night because of work stress.",
    ),
    _segment(
        18,
        "SPEAKER_00",
        "I think your headaches may be related to elevated blood pressure and poor sleep. I'd like to start lisinopril ten milligrams daily and have you monitor your blood pressure at home.",
    ),
    _segment(19, "SPEAKER_01", "Okay, that sounds reasonable."),
    _segment(
        20,
        "SPEAKER_00",
        "We'll also order some basic blood work and schedule a follow-up visit in four weeks.",
    ),
    _segment(21, "SPEAKER_01", "Sounds good. Thank you, doctor."),
    _segment(
        22,
        "SPEAKER_00",
        "You're welcome. Let us know if the headaches worsen or if you develop any new symptoms.",
    ),
]

result = generate_soap(segments, specialty="primary_care")
warnings = validate_soap(result)

print("Chief complaint:", result.chief_complaint)
print("Subjective:", result.subjective)
print("Objective:", result.objective)
print("Assessment:", result.assessment)
print("Plan:", result.plan)
print("ICD-10 suggestions:", result.icd10_suggestions)
print("Raw claims:", result.raw_claims)
print("Warnings:", warnings)
print(f"Generation latency: {result.generation_ms:.0f} ms")

assert result.subjective.strip()
assert result.objective.strip()
assert result.assessment.strip()
assert result.plan.strip()
assert result.icd10_suggestions
assert result.raw_claims
assert not any(
    "post-traumatic" in suggestion.lower() or "G44.309" in suggestion
    for suggestion in result.icd10_suggestions
)
assert not warnings
