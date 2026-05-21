from __future__ import annotations

import pytest

from fishs2_fastapi.api_models import CreateSpeechRequest
from fishs2_fastapi.engine import engine
from fishs2_fastapi.errors import APIError
from fishs2_fastapi.registry import ModelInfo


def test_generate_default_voice_without_reference(monkeypatch):
    class FakeRuntime:
        def synthesize_to_wav_bytes(self, **kwargs):
            return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr(engine, "_get_runtime", lambda backend_model_id: FakeRuntime())

    request = CreateSpeechRequest(
        model="fishs2",
        input="Hello",
        voice="default",
    )
    model_info = ModelInfo(model_id="fishs2", backend_model_id="fishaudio/s2-pro")

    audio = engine.generate_speech(request, model_info)
    assert audio[:4] == b"RIFF"


def test_generate_rejects_speed_override(monkeypatch):
    class FakeRuntime:
        def synthesize_to_wav_bytes(self, **kwargs):
            return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr(engine, "_get_runtime", lambda backend_model_id: FakeRuntime())

    request = CreateSpeechRequest(
        model="fishs2",
        input="Hello",
        voice="default",
        speed=1.25,
    )
    model_info = ModelInfo(model_id="fishs2", backend_model_id="fishaudio/s2-pro")

    with pytest.raises(APIError) as exc:
        engine.generate_speech(request, model_info)

    assert exc.value.code == "unsupported_speed"


def test_generate_requires_transcript_when_reference_audio_present(monkeypatch):
    class FakeRuntime:
        def synthesize_to_wav_bytes(self, **kwargs):
            return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr(engine, "_get_runtime", lambda backend_model_id: FakeRuntime())

    request = CreateSpeechRequest(
        model="fishs2",
        input="Hello",
        voice="default",
        reference_audio="E:/voices/ref.wav",
    )
    model_info = ModelInfo(model_id="fishs2", backend_model_id="fishaudio/s2-pro")

    with pytest.raises(APIError) as exc:
        engine.generate_speech(request, model_info)

    assert exc.value.code == "missing_prompt_text"


def test_generate_uses_reference_text_field(monkeypatch):
    captured = {}

    class FakeRuntime:
        def synthesize_to_wav_bytes(self, **kwargs):
            captured.update(kwargs)
            return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr(engine, "_get_runtime", lambda backend_model_id: FakeRuntime())

    request = CreateSpeechRequest(
        model="fishs2",
        input="Hello",
        voice="default",
        reference_audio="E:/voices/ref.wav",
        reference_text="Reference transcript",
    )
    model_info = ModelInfo(model_id="fishs2", backend_model_id="fishaudio/s2-pro")

    audio = engine.generate_speech(request, model_info)
    assert audio[:4] == b"RIFF"
    assert captured["reference_audio_path"] == "E:/voices/ref.wav"
    assert captured["reference_text"] == "Reference transcript"
