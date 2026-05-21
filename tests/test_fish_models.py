from fastapi.testclient import TestClient

from fishs2_fastapi.main import app
from fishs2_fastapi.registry import ModelRegistry

client = TestClient(app)


def test_list_models_contains_fish_entries():
    resp = client.get("/v1/models")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["object"] == "list"
    model_ids = [item["id"] for item in payload["data"]]
    assert "fishaudio/s2-pro" in model_ids
    assert "fishs2" in model_ids


def test_registry_maps_alias_to_default_backend():
    local_registry = ModelRegistry()
    local_registry.discover()

    canonical = local_registry.get("fishaudio/s2-pro")
    alias = local_registry.get("fishs2")

    assert canonical is not None
    assert alias is not None
    assert alias.backend_model_id == canonical.backend_model_id == "fishaudio/s2-pro"


def test_speech_rejects_unknown_model():
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "nonexistent_model",
            "input": "Hello",
            "voice": "default",
        },
    )

    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == "model_not_found"
