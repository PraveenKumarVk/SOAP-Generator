# test_asr.py — run this before moving to Phase 2
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    print(f"Re-running with project venv: {VENV_PYTHON}", flush=True)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(PROJECT_ROOT))
from backend.pipeline.asr import WhisperXProvider

import json

provider = WhisperXProvider()

result = provider.transcribe("demo_audio/doc-conversation.wav")

print(f"Language: {result.language}")

print(f"Segments: {len(result.segments)}")

print(f"Speakers found: {set(s.speaker for s in result.segments)}")

print(f"Latency: {result.latency}")

# Must pass before Phase 2:

assert len(result.segments) > 0

assert any(s.speaker != "" for s in result.segments)

assert result.latency.total_ms > 0
