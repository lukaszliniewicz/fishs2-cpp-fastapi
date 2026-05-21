from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from voxcpm_fastapi.main import app

client = TestClient(app)


def test_list_voices_aliases_match():
    primary = client.get("/v1/audio/voices")
    assert primary.status_code == 200
    assert primary.json()["object"] == "list"

    alias = client.get("/v1/voices")
    assert alias.status_code == 200
    assert alias.json() == primary.json()

    legacy = client.get("/v1/files")
    assert legacy.status_code == 200
    assert legacy.json() == primary.json()


def test_create_and_delete_voice_with_files_field():
    voice_id = f"voice-{uuid4().hex}"
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 512)

    created = client.post(
        "/v1/audio/voices",
        files={"files": ("sample.wav", payload, "audio/wav")},
        data={"voice_id": voice_id},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["id"] == voice_id
    assert body["sample_count"] == 1

    listed = client.get("/v1/audio/voices")
    assert listed.status_code == 200
    ids = [item["voice_id"] for item in listed.json()["data"]]
    assert voice_id in ids

    deleted = client.delete(f"/v1/voices/{voice_id}")
    assert deleted.status_code == 200


def test_create_voice_accepts_audio_sample_field():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 256)

    created = client.post(
        "/v1/audio/voices",
        files={"audio_sample": ("My Cool Voice.wav", payload, "audio/wav")},
    )
    assert created.status_code == 200
    data = created.json()
    assert data["id"] == "my-cool-voice"

    client.delete(f"/v1/voices/{data['id']}")


def test_create_voice_hifi_metadata_persists():
    voice_id = f"hifi-{uuid4().hex}"
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 256)

    created = client.post(
        "/v1/audio/voices",
        files={"files": ("hifi.wav", payload, "audio/wav")},
        data={
            "voice_id": voice_id,
            "mode": "hifi",
            "prompt_text": "Reference transcript.",
        },
    )
    assert created.status_code == 200
    assert created.json()["mode"] == "hifi"

    listed = client.get("/v1/audio/voices")
    assert listed.status_code == 200
    match = next((item for item in listed.json()["data"] if item["voice_id"] == voice_id), None)
    assert match is not None
    assert match["mode"] == "hifi"
    assert match["prompt_text"] == "Reference transcript."

    client.delete(f"/v1/voices/{voice_id}")


def test_create_voice_rejects_invalid_mode():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 128)
    created = client.post(
        "/v1/audio/voices",
        files={"files": ("sample.wav", payload, "audio/wav")},
        data={"mode": "unknown"},
    )

    assert created.status_code == 422
    assert created.json()["error"]["code"] == "invalid_mode"
