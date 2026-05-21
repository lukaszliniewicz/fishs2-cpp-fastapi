from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxcpm_fastapi.api_models import CreateSpeechRequest, VoxCPMParams
from voxcpm_fastapi.engine import engine
from voxcpm_fastapi.errors import APIError
from voxcpm_fastapi.registry import ModelInfo


def test_build_generate_kwargs_applies_voxcpm_overrides():
    request = CreateSpeechRequest(
        model="voxcpm2",
        input="Hello",
        voice="default",
        voxcpm=VoxCPMParams(cfg_value=3.0, inference_timesteps=14, retry_badcase=False),
    )

    kwargs = engine._build_generate_kwargs(
        request,
        reference_wav_path=None,
        prompt_wav_path=None,
        prompt_text=None,
    )

    assert kwargs["cfg_value"] == pytest.approx(3.0)
    assert kwargs["inference_timesteps"] == 14
    assert kwargs["retry_badcase"] is False


def test_build_generate_kwargs_applies_control_prefix():
    request = CreateSpeechRequest(
        model="voxcpm2",
        input="Hello",
        voice="default",
        control="warm female voice",
    )

    kwargs = engine._build_generate_kwargs(
        request,
        reference_wav_path=None,
        prompt_wav_path=None,
        prompt_text=None,
    )
    assert kwargs["text"].startswith("(warm female voice)")


def test_generate_speech_hifi_requires_prompt_text(monkeypatch):
    request = CreateSpeechRequest(
        model="voxcpm2",
        input="Hello",
        voice="default",
        mode="hifi",
    )
    model_info = ModelInfo(model_id="voxcpm2", backend_model_id="openbmb/VoxCPM2")

    with pytest.raises(APIError) as exc:
        engine.generate_speech(request, model_info)

    assert exc.value.code == "missing_reference_audio"


def test_generate_speech_rejects_short_reference_audio(tmp_path, monkeypatch):
    short_wav = tmp_path / "short.wav"
    sf.write(short_wav, np.zeros(1200, dtype=np.float32), 48000)

    monkeypatch.setattr("voxcpm_fastapi.engine.settings.min_reference_audio_seconds", 0.1)
    request = CreateSpeechRequest(
        model="voxcpm2",
        input="Hello",
        voice="default",
        speaker_wav=[str(short_wav)],
    )

    model_info = ModelInfo(model_id="voxcpm2", backend_model_id="openbmb/VoxCPM2")
    with pytest.raises(APIError) as exc:
        engine.generate_speech(request, model_info)

    assert exc.value.code == "reference_audio_too_short"


def test_generate_speech_uses_wrapper_and_returns_wav(monkeypatch, tmp_path):
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, np.zeros(4800, dtype=np.float32), 48000)

    class FakeWrapper:
        sample_rate = 48000

        def generate(self, **kwargs):
            return np.array([0.0, 0.1, -0.1], dtype=np.float32)

    monkeypatch.setattr(engine, "_get_wrapper", lambda backend_model_id: FakeWrapper())
    monkeypatch.setattr("voxcpm_fastapi.engine.settings.min_reference_audio_seconds", 0.0)

    request = CreateSpeechRequest(
        model="voxcpm2",
        input="Hello",
        voice="default",
        speaker_wav=[str(wav_path)],
    )
    model_info = ModelInfo(model_id="voxcpm2", backend_model_id="openbmb/VoxCPM2")

    audio = engine.generate_speech(request, model_info)
    assert audio[:4] == b"RIFF"
