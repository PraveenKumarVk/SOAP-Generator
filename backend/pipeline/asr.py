from __future__ import annotations

import time
from datetime import datetime
from typing import Protocol

from backend import config
from backend.models import LatencyBreakdown, Segment, TranscriptionResult, Word


class Provider(Protocol):
    def transcribe(self, audio_path: str) -> TranscriptionResult: ...


class WhisperXProvider:
    def __init__(self) -> None:
        _log(
            "loading ASR model "
            f"engine={config.ASR_ENGINE} "
            f"model={config.WHISPER_MODEL_SIZE} device={config.DEVICE} "
            f"compute={config.COMPUTE_TYPE} language={config.LANGUAGE or 'auto'}"
        )
        if config.ASR_ENGINE == "openai-whisper":
            import whisper

            self.model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=config.DEVICE)
        elif config.ASR_ENGINE == "whisperx":
            whisperx = _load_whisperx()
            self.model = whisperx.load_model(
                config.WHISPER_MODEL_SIZE,
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                language=config.LANGUAGE or None,
            )
        else:
            raise ValueError(
                f"Unknown ASR_ENGINE: {config.ASR_ENGINE!r}. "
                "Choose 'openai-whisper' or 'whisperx'."
            )
        _log("ASR model loaded")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        t0 = time.perf_counter()

        _log("running ASR transcription")
        result = self._transcribe_audio(audio_path)
        language = result.get("language") or config.LANGUAGE or "en"
        t1 = time.perf_counter()
        _log(f"ASR complete in {_ms(t0, t1):.0f} ms; language={language}")

        if config.ASR_ENGINE == "openai-whisper":
            diarization_ms = 0.0

            if config.HF_TOKEN:
                _log("loading diarization model")
                try:
                    from pyannote.audio import Pipeline as PyannotePipeline

                    t_diarize_start = time.perf_counter()

                    diarize_pipeline = PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.0",
                        use_auth_token=config.HF_TOKEN,
                    )
                    diarize_pipeline.to(__import__("torch").device(config.DEVICE))

                    _log("running diarization")
                    diarization = diarize_pipeline(
                        audio_path,
                        num_speakers=2,
                    )

                    _log("assigning speakers to segments")
                    for seg in result["segments"]:
                        seg["speaker"] = _get_speaker_for_segment(
                            seg.get("start", 0.0),
                            seg.get("end", 0.0),
                            diarization,
                        )

                    diarization_ms = _ms(t_diarize_start, time.perf_counter())
                    _log(f"diarization complete in {diarization_ms:.0f} ms")

                except Exception as exc:
                    _log(f"diarization failed: {exc} — continuing without speakers")
                    diarization_ms = 0.0
            else:
                _log("HF_TOKEN not set — skipping diarization")
                diarization_ms = 0.0

            segments = [
                Segment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker", ""),
                    words=[
                        Word(
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            word=w.get("word", ""),
                            score=w.get("score"),
                        )
                        for w in seg.get("words", [])
                    ],
                )
                for seg in result.get("segments", [])
            ]
            latency = LatencyBreakdown(
                load_audio_ms=0.0,
                asr_ms=_ms(t0, t1),
                alignment_ms=0.0,
                diarization_ms=diarization_ms,
                total_ms=_ms(t0, t1) + diarization_ms,
            )
            return TranscriptionResult(
                language=language,
                segments=segments,
                latency=latency,
                pipeline_used=config.ASR_ENGINE,
            )

        whisperx = _load_whisperx()

        _log(f"loading audio {audio_path}")
        audio = whisperx.load_audio(audio_path)
        t2 = time.perf_counter()
        _log(f"audio loaded in {_ms(t1, t2):.0f} ms")

        _log("loading alignment model")
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=config.DEVICE
        )
        _log("running word alignment")
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device=config.DEVICE
        )
        t3 = time.perf_counter()
        _log(f"alignment complete in {_ms(t2, t3):.0f} ms")

        _log("loading diarization model")
        if not config.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is required for pyannote diarization. Add it to the project "
                ".env file, then accept access terms for pyannote/speaker-diarization-3.1 "
                "on Hugging Face."
            )

        try:
            diarize_model = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.0",
                use_auth_token=config.HF_TOKEN,
                device=config.DEVICE,
            )
        except Exception as exc:
            if _looks_like_pyannote_access_or_download_error(exc):
                raise RuntimeError(_pyannote_access_message()) from exc
            raise

        if diarize_model is None:
            raise RuntimeError(_pyannote_access_message())

        _log("running diarization")
        try:
            diarize_segments = diarize_model(audio_path, num_speakers=2)
        except AttributeError as exc:
            if _looks_like_pyannote_access_or_download_error(exc):
                raise RuntimeError(_pyannote_access_message()) from exc
            raise

        _log("assigning speakers to words")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        t4 = time.perf_counter()
        _log(f"diarization complete in {_ms(t3, t4):.0f} ms")

        latency = LatencyBreakdown(
            load_audio_ms=_ms(t1, t2),
            asr_ms=_ms(t0, t1),
            alignment_ms=_ms(t2, t3),
            diarization_ms=_ms(t3, t4),
            total_ms=_ms(t0, t4),
        )

        segments = [
            Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker", ""),
                words=[
                    Word(
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        word=w.get("word", ""),
                        score=w.get("score"),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in result["segments"]
        ]

        return TranscriptionResult(
            language=language,
            segments=segments,
            latency=latency,
            pipeline_used=config.ASR_ENGINE,
        )

    def _transcribe_audio(self, audio_path: str) -> dict:
        if config.ASR_ENGINE == "openai-whisper":
            return self.model.transcribe(
                audio_path,
                language=config.LANGUAGE or None,
                fp16=False,
                verbose=True,
            )

        whisperx = _load_whisperx()
        audio = whisperx.load_audio(audio_path)
        return self.model.transcribe(
            audio,
            batch_size=1,
            language=config.LANGUAGE or None,
            print_progress=True,
        )


class LFMProvider:
    """
    LFM 2.5-Audio provider. Requires the `liquid-audio` package.
    Sets self.available = False gracefully when the package is absent.
    """
    def __init__(self) -> None:
        try:
            import liquid_audio  # type: ignore[import]
            self._lib = liquid_audio
            self.available = True
            _log("LFM2.5-Audio loaded")
        except ImportError:
            self._lib = None
            self.available = False
            _log("liquid-audio not installed — LFMProvider unavailable")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        if not self.available:
            raise NotImplementedError(
                "LFM2.5-Audio: install liquid-audio package. "
                "See README section 'LFM Integration'."
            )
        # Real implementation goes here once liquid-audio API is confirmed.
        raise NotImplementedError("LFM2.5-Audio transcribe() — implementation pending")


_HYBRID_CONFIDENCE_THRESHOLD = 0.70


def _mean_word_confidence(result: TranscriptionResult) -> float:
    """Average word-level score across all segments; 1.0 when no scores present."""
    scores = [
        w.score
        for seg in result.segments
        for w in seg.words
        if w.score is not None
    ]
    return sum(scores) / len(scores) if scores else 1.0


class HybridProvider:
    """
    Confidence-routing provider: runs WhisperX by default.
    When LFM is available and WhisperX mean word-confidence falls below
    _HYBRID_CONFIDENCE_THRESHOLD, re-routes to LFM for a second pass.
    Falls back to WhisperX entirely if LFM is unavailable.
    """
    def __init__(self) -> None:
        self._lfm = LFMProvider()
        self._whisperx = WhisperXProvider()

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        if not self._lfm.available:
            _log("HybridProvider: LFM unavailable — routing to WhisperX")
            return self._whisperx.transcribe(audio_path)

        whisperx_result = self._whisperx.transcribe(audio_path)
        mean_conf = _mean_word_confidence(whisperx_result)

        if mean_conf >= _HYBRID_CONFIDENCE_THRESHOLD:
            _log(
                f"HybridProvider: WhisperX confidence {mean_conf:.2f} ≥ "
                f"{_HYBRID_CONFIDENCE_THRESHOLD} — using WhisperX"
            )
            return whisperx_result

        _log(
            f"HybridProvider: WhisperX confidence {mean_conf:.2f} < "
            f"{_HYBRID_CONFIDENCE_THRESHOLD} — routing to LFM"
        )
        return self._lfm.transcribe(audio_path)


def route_pipeline(pipeline: str) -> Provider:
    if pipeline in ("whisper", "whisperx", "openai-whisper"):
        return WhisperXProvider()
    if pipeline == "lfm":
        return LFMProvider()
    if pipeline == "hybrid":
        return HybridProvider()
    raise ValueError(
        f"Unknown pipeline: {pipeline!r}. Choose 'whisper', 'lfm', or 'hybrid'."
    )


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _ms(a: float, b: float) -> float:
    return (b - a) * 1000.0


def _load_whisperx():
    import whisperx

    return whisperx


def _get_speaker_for_segment(start: float, end: float, diarization) -> str:
    """Assign speaker by maximum overlap with diarized segments."""
    best_speaker = ""
    best_overlap = 0.0
    try:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(start, turn.start)
            overlap_end = min(end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
    except Exception:
        return ""
    return best_speaker


def _pyannote_access_message() -> str:
    return (
        "Could not load pyannote diarization. Your HF_TOKEN is set, but the Hugging Face "
        "account for that token must have read access and must accept the model terms for "
        "https://huggingface.co/pyannote/speaker-diarization-3.1 , "
        "https://huggingface.co/pyannote/speaker-diarization-3.0 , and "
        "https://huggingface.co/pyannote/segmentation-3.0 . If access was just granted, "
        "create or reuse a read token from the same account and rerun the test."
    )


def _looks_like_pyannote_access_or_download_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        detail = f"{type(current).__module__}.{type(current).__name__}: {current}"
        if any(
            text in detail
            for text in (
                "'NoneType' object has no attribute 'to'",
                "'NoneType' object has no attribute 'eval'",
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization-3.0",
                "pyannote/segmentation-3.0",
                "huggingface.co",
                "huggingface_hub",
                "LocalEntryNotFoundError",
                "Could not download",
                "gated",
                "private",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
