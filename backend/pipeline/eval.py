from __future__ import annotations

import json
import re
import warnings

import anthropic
import jiwer
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

from backend import config
from backend.db import save_metrics
from backend.models import (
    EncounterMetrics,
    HallucinationFlag,
    HallucinationResult,
    LatencyBreakdown,
    MedicalWERResult,
    PDQIResult,
    PDQIScores,
    Segment,
    SOAPResult,
)

HALLUCINATION_THRESHOLD = 0.45

# Suppress spaCy FutureWarning from en_core_sci_lg tokenizer deserializer
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")

_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_ner_bc5cdr_md")
    return _nlp


# en_ner_bc5cdr_md produces CHEMICAL (medications) and DISEASE (symptoms).
# Procedures have no label in this model; empty list triggers the None guard.
ENTITY_CATEGORY_MAP: dict[str, list[str]] = {
    "medications": ["CHEMICAL"],
    "symptoms": ["DISEASE"],
    "procedures": [],
}


def compute_medical_wer(reference: str, hypothesis: str) -> MedicalWERResult:
    nlp = _get_nlp()
    ref_doc = nlp(reference)
    hyp_doc = nlp(hypothesis)

    entity_counts: dict[str, int] = {}
    results: dict[str, float | None] = {}

    for category, labels in ENTITY_CATEGORY_MAP.items():
        if not labels:
            results[category] = None
            entity_counts[category] = 0
            continue

        ref_ents = [ent.text.lower() for ent in ref_doc.ents if ent.label_ in labels]
        hyp_ents = [ent.text.lower() for ent in hyp_doc.ents if ent.label_ in labels]
        entity_counts[category] = len(ref_ents)
        results[category] = _safe_wer(ref_ents, hyp_ents)  # None if ref is empty

    # Overall: every entity in the doc regardless of label
    ref_all = [ent.text.lower() for ent in ref_doc.ents]
    hyp_all = [ent.text.lower() for ent in hyp_doc.ents]
    entity_counts["overall"] = len(ref_all)
    overall_medical = _safe_wer(ref_all, hyp_all)

    return MedicalWERResult(
        medications=results["medications"],
        symptoms=results["symptoms"],
        procedures=results["procedures"],
        overall_medical=overall_medical,
        entity_counts=entity_counts,
    )


_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def check_hallucinations(
    soap_result: SOAPResult,
    segments: list[Segment],
) -> HallucinationResult:
    claims = list(soap_result.note.raw_claims)
    if not claims:
        claims = _extract_claims_via_claude(soap_result)

    # Build per-sentence source lookup: text → speaker
    transcript_sentences: list[str] = [seg.text.strip() for seg in segments if seg.text.strip()]
    speaker_lookup: dict[str, str] = {
        seg.text.strip(): seg.speaker
        for seg in segments
        if seg.text.strip()
    }

    # Edge case: no transcript to ground against
    if not transcript_sentences:
        flags = [
            HallucinationFlag(
                claim=c,
                grounded=False,
                max_similarity=0.0,
                best_source_text="",
                best_source_speaker="",
            )
            for c in claims
        ]
        return HallucinationResult(
            flags=flags,
            grounded_count=0,
            ungrounded_count=len(flags),
            threshold_used=HALLUCINATION_THRESHOLD,
        )

    model = _get_st_model()
    transcript_embs = model.encode(transcript_sentences, convert_to_tensor=True)

    flags: list[HallucinationFlag] = []
    for claim in claims:
        claim_emb = model.encode([claim], convert_to_tensor=True)
        sims = util.cos_sim(claim_emb, transcript_embs)[0].cpu().numpy()
        argmax = int(np.argmax(sims))
        max_sim = float(sims[argmax])
        best_text = transcript_sentences[argmax]

        flags.append(
            HallucinationFlag(
                claim=claim,
                grounded=max_sim >= HALLUCINATION_THRESHOLD,
                max_similarity=round(max_sim, 4),
                best_source_text=best_text,
                best_source_speaker=speaker_lookup.get(best_text, ""),
            )
        )

    grounded = sum(1 for f in flags if f.grounded)
    return HallucinationResult(
        flags=flags,
        grounded_count=grounded,
        ungrounded_count=len(flags) - grounded,
        threshold_used=HALLUCINATION_THRESHOLD,
    )


def _extract_claims_via_claude(soap_result: SOAPResult) -> list[str]:
    ap_text = (
        f"Assessment: {soap_result.note.assessment}\n\n"
        f"Plan: {soap_result.note.plan}"
    )
    client = anthropic.Anthropic(api_key=config.require_anthropic_api_key())
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{ap_text}\n\n"
                    "Extract discrete factual claims as a JSON array of strings. "
                    "Claims only, no meta-commentary."
                ),
            }
        ],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# PDQI proxy
# ---------------------------------------------------------------------------

_PDQI_DISCLAIMER = (
    "LLM-proxy rubric adapted from PDQI-9 (Stetson et al., 2012). "
    "Not validated against human clinical raters. "
    "Use for relative comparison only."
)

_PDQI_DIMENSIONS = ("accuracy", "completeness", "organization", "conciseness", "attribution")

_PDQI_SYSTEM = (
    "You are evaluating a clinical SOAP note generated by an AI scribe. "
    "Score each dimension 1-5. "
    "Think step by step before each score. "
    "Return ONLY valid JSON matching the exact schema provided."
)

_PDQI_USER_TEMPLATE = """\
Evaluate this AI-generated SOAP note against the source transcript.

TRANSCRIPT:
{transcript}

SOAP NOTE:
{soap_note}

Score each dimension 1–5 (5 = best) and briefly explain:
- accuracy: Does the note accurately reflect what was discussed?
- completeness: Are all clinically relevant details captured?
- organization: Is the note well-structured and logically ordered?
- conciseness: Is the note concise without losing key information?
- attribution: Are physician vs patient statements correctly attributed?

Think step by step before each score, then return ONLY this JSON schema — \
no other text, no markdown:
{{
  "scores": {{
    "accuracy": <int 1-5>,
    "completeness": <int 1-5>,
    "organization": <int 1-5>,
    "conciseness": <int 1-5>,
    "attribution": <int 1-5>
  }},
  "reasoning": {{
    "accuracy": "<explanation>",
    "completeness": "<explanation>",
    "organization": "<explanation>",
    "conciseness": "<explanation>",
    "attribution": "<explanation>"
  }}
}}
"""


def compute_pdqi_proxy(transcript: str, soap_note: dict) -> PDQIResult:
    client = anthropic.Anthropic(api_key=config.require_anthropic_api_key())
    user_prompt = _PDQI_USER_TEMPLATE.format(
        transcript=transcript,
        soap_note=json.dumps(soap_note, indent=2),
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        temperature=0,
        system=_PDQI_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"PDQI response is not valid JSON: {exc}\n\nRaw:\n{raw}"
        ) from exc

    scores_raw: dict = data.get("scores", {})
    reasoning_raw: dict = data.get("reasoning", {})

    scores = PDQIScores(
        accuracy=int(scores_raw.get("accuracy", 3)),
        completeness=int(scores_raw.get("completeness", 3)),
        organization=int(scores_raw.get("organization", 3)),
        conciseness=int(scores_raw.get("conciseness", 3)),
        attribution=int(scores_raw.get("attribution", 3)),
    )
    reasoning = {dim: str(reasoning_raw.get(dim, "")) for dim in _PDQI_DIMENSIONS}
    mean_score = sum(getattr(scores, d) for d in _PDQI_DIMENSIONS) / len(_PDQI_DIMENSIONS)

    return PDQIResult(
        scores=scores,
        reasoning=reasoning,
        mean_score=round(mean_score, 2),
        disclaimer=_PDQI_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Full evaluation orchestrator
# ---------------------------------------------------------------------------

def run_full_eval(
    encounter_id: str,
    reference_transcript: str | None,
    hypothesis_transcript: str,
    soap_result: SOAPResult,
    segments: list[Segment],
    latency: LatencyBreakdown | None = None,
) -> EncounterMetrics:
    import logging
    log = logging.getLogger(__name__)

    medical_wer: MedicalWERResult | None = None
    hallucination: HallucinationResult | None = None
    pdqi: PDQIResult | None = None

    if reference_transcript is not None:
        try:
            medical_wer = compute_medical_wer(reference_transcript, hypothesis_transcript)
        except Exception as exc:
            log.warning("medical_wer failed: %s", exc)

    try:
        hallucination = check_hallucinations(soap_result, segments)
    except Exception as exc:
        log.warning("hallucination check failed: %s", exc)

    try:
        transcript_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        pdqi = compute_pdqi_proxy(transcript_text, soap_result.model_dump())
    except Exception as exc:
        log.warning("pdqi_proxy failed: %s", exc)

    lat = latency or LatencyBreakdown()
    metrics_dict: dict = {
        "vad_ms": lat.load_audio_ms,
        "asr_ms": lat.asr_ms + lat.alignment_ms,
        "diarization_ms": lat.diarization_ms,
        "note_gen_ms": soap_result.generation_ms,
        "total_ms": lat.total_ms + soap_result.generation_ms,
        "medical_wer_medications": medical_wer.medications if medical_wer else None,
        "medical_wer_symptoms": medical_wer.symptoms if medical_wer else None,
        "medical_wer_procedures": medical_wer.procedures if medical_wer else None,
        "hallucination_flags": (
            [f.model_dump() for f in hallucination.flags] if hallucination else None
        ),
        "hallucination_count": hallucination.ungrounded_count if hallucination else None,
        "pdqi_scores": pdqi.scores.model_dump() if pdqi else None,
        "pdqi_mean": pdqi.mean_score if pdqi else None,
    }

    save_metrics(encounter_id, metrics_dict)

    return EncounterMetrics(
        encounter_id=encounter_id,
        latency=lat,
        medical_wer=medical_wer,
        hallucination=hallucination,
        pdqi=pdqi,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_wer(ref_ents: list[str], hyp_ents: list[str]) -> float | None:
    if not ref_ents:
        return None  # cannot compute WER with empty reference
    ref_str = " ".join(ref_ents)
    hyp_str = " ".join(hyp_ents)
    if not hyp_str:
        return 1.0  # all reference entities missed
    return jiwer.wer(ref_str, hyp_str)
