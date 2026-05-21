from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from voxcpm_fastapi.main import app

client = TestClient(app)


def test_legacy_files_post_accepts_file_field():
    voice_id = f"legacy-{uuid4().hex}"
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 256)

    created = client.post(
        "/v1/files",
        files={"file": ("legacy.wav", payload, "audio/wav")},
        data={"voice_id": voice_id, "purpose": "user_data"},
    )
    assert created.status_code == 200

    body = created.json()
    assert body["id"] == voice_id
    assert body["object"] == "voice"

    listed = client.get("/v1/files")
    assert listed.status_code == 200
    ids = [item["voice_id"] for item in listed.json()["data"]]
    assert voice_id in ids

    client.delete(f"/v1/voices/{voice_id}")


def test_legacy_files_post_accepts_xtts_style_files_field():
    payload = b"RIFF\x00\x00\x00\x00WAVE" + (b"\x00" * 256)

    created = client.post(
        "/v1/files",
        files={"files": ("Team Voice.wav", payload, "audio/wav")},
        data={"purpose": "assistants"},
    )
    assert created.status_code == 200

    voice_id = created.json()["id"]
    assert voice_id == "team-voice"
    client.delete(f"/v1/voices/{voice_id}")
