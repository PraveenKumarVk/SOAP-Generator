from __future__ import annotations

import json
import re
import time

import anthropic

from backend import config
from backend.models import Segment, SOAPNote, SOAPResult

_REQUIRED_KEYS = {"S", "O", "A", "P", "chief_complaint", "icd10_suggestions", "raw_claims"}

_SPEAKER_LABELS: dict[str, str] = {
    "SPEAKER_00": "Physician",
    "SPEAKER_01": "Patient",
}

_SPECIALTY_PROMPTS: dict[str, str] = {
    "primary_care": (
        "You are a clinical documentation assistant for a primary care physician. "
        "Your task is to convert a doctor-patient conversation into a structured SOAP note. "
        "Prioritise: complete medication reconciliation, preventive care items, "
        "chronic disease management, and social determinants of health in the Plan. "
        "Keep language concise and medically precise. "
        "Return ONLY valid JSON — no markdown, no explanation, no preamble."
    ),
    "cardiology": (
        "You are a clinical documentation assistant for a cardiologist. "
        "Your task is to convert a doctor-patient conversation into a structured SOAP note. "
        "Prioritise: haemodynamic findings, rhythm and rate, current cardiac medications, "
        "device history (pacemaker/ICD), risk stratification, and cardiology-specific plan items "
        "such as imaging orders, electrophysiology referrals, and anticoagulation management. "
        "Keep language concise and medically precise. "
        "Return ONLY valid JSON — no markdown, no explanation, no preamble."
    ),
}

_USER_PROMPT_TEMPLATE = """\
Convert the following clinical conversation into a SOAP note.

TRANSCRIPT:
{transcript}

Output ONLY a single valid JSON object with exactly these keys:
{{
  "S": "<Subjective: patient's chief complaint, symptoms, history in their own words>",
  "O": "<Objective: vitals, exam findings, lab/imaging results mentioned>",
  "A": "<Assessment: diagnosis or differential>",
  "P": "<Plan: treatments, medications, follow-up, referrals>",
  "chief_complaint": "<one-sentence chief complaint>",
  "icd10_suggestions": ["<CODE: description>", ...],
  "raw_claims": ["<discrete factual claim from A or P>", ...]
}}

Rules:
- icd10_suggestions: maximum 3 ICD-10 codes, each as "CODE: description"
- icd10_suggestions must be supported by the transcript; do not suggest diagnoses
  for events, causes, or histories that were not mentioned
- For headache complaints, use R51.9 unless the transcript explicitly supports a
  more specific headache subtype; do not use G44.* codes without explicit support
- raw_claims: list every discrete factual claim made in the Assessment and Plan \
sections (used for hallucination checking); do not leave this empty
- Do not include any text outside the JSON object
"""


def generate_soap(
    segments: list[Segment],
    specialty: str = "primary_care",
) -> SOAPResult:
    if specialty not in _SPECIALTY_PROMPTS:
        raise ValueError(
            f"Unknown specialty: {specialty!r}. "
            f"Choose one of: {sorted(_SPECIALTY_PROMPTS)}"
        )

    transcript = _build_transcript(segments)
    system_prompt = _SPECIALTY_PROMPTS[specialty]
    user_prompt = _USER_PROMPT_TEMPLATE.format(transcript=transcript)

    api_key = config.require_anthropic_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    t0 = time.perf_counter()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    generation_ms = (time.perf_counter() - t0) * 1000.0

    raw_text = message.content[0].text
    parsed = _parse_response(raw_text)

    note = SOAPNote(
        subjective=parsed["S"],
        objective=parsed["O"],
        assessment=parsed["A"],
        plan=parsed["P"],
        chief_complaint=parsed["chief_complaint"],
        icd10_suggestions=parsed["icd10_suggestions"],
        raw_claims=parsed["raw_claims"],
    )
    return SOAPResult(note=note, generation_ms=generation_ms, specialty=specialty)


def validate_soap(result: SOAPResult) -> list[str]:
    warnings: list[str] = []

    sections = {
        "subjective": result.note.subjective,
        "objective": result.note.objective,
        "assessment": result.note.assessment,
        "plan": result.note.plan,
    }
    for name, text in sections.items():
        if len(text.strip()) < 20:
            warnings.append(f"Section '{name}' is suspiciously short ({len(text.strip())} chars)")

    if len(result.note.icd10_suggestions) > 5:
        warnings.append(
            f"Too many ICD-10 suggestions ({len(result.note.icd10_suggestions)}); max recommended is 5"
        )

    if not result.note.raw_claims:
        warnings.append("raw_claims is empty — hallucination evaluation will fail silently")

    return warnings


def _build_transcript(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        label = _SPEAKER_LABELS.get(seg.speaker, "Unknown")
        lines.append(f"{label}: {seg.text.strip()}")
    return "\n".join(lines)


def _parse_response(raw_text: str) -> dict:
    text = raw_text.strip()

    # Strip accidental markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SOAP generation returned invalid JSON: {exc}\n\nRaw response:\n{raw_text}"
        ) from exc

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(
            f"SOAP JSON missing required keys: {missing}\n\nRaw response:\n{raw_text}"
        )

    return data
