import pytest
from fastapi.testclient import TestClient

from voxcpm_fastapi.main import app

client = TestClient(app)


def test_speech_unsupported_format():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello",
            "voice": "default",
            "response_format": "mp3",
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "unsupported_format"


def test_speech_instructions_invalid_json():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello",
            "voice": "default",
            "instructions": "{not json",
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "invalid_instructions_json"


def test_speech_instructions_invalid_voxcpm_payload():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello",
            "voice": "default",
            "instructions": '{"voxcpm": {"inference_timesteps": -1}}',
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "invalid_instructions_voxcpm"


def test_speech_instructions_apply_voxcpm_overrides(monkeypatch):
    captured = {}

    async def fake_generate(request, model_info=None):
        captured["cfg_value"] = request.voxcpm.cfg_value if request.voxcpm else None
        captured["timesteps"] = request.voxcpm.inference_timesteps if request.voxcpm else None
        return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("voxcpm_fastapi.main.engine.generate_speech_async", fake_generate)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello world",
            "voice": "default",
            "instructions": '{"voxcpm": {"cfg_value": 3.5}, "inference_timesteps": 12}',
        },
    )

    assert resp.status_code == 200
    assert captured["cfg_value"] == pytest.approx(3.5)
    assert captured["timesteps"] == 12


def test_speech_instructions_apply_scalar_overrides(monkeypatch):
    captured = {}

    async def fake_generate(request, model_info=None):
        captured["mode"] = request.mode
        captured["prompt_text"] = request.prompt_text
        captured["control"] = request.control
        return b"RIFF" + (b"\x00" * 64)

    monkeypatch.setattr("voxcpm_fastapi.main.engine.generate_speech_async", fake_generate)

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello world",
            "voice": "default",
            "instructions": '{"mode": "hifi", "prompt_text": "reference line", "control": "warm voice"}',
        },
    )

    assert resp.status_code == 200
    assert captured["mode"] == "hifi"
    assert captured["prompt_text"] == "reference line"
    assert captured["control"] == "warm voice"


def test_speech_unknown_model():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "unknown/model",
            "input": "Hello",
            "voice": "default",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"
