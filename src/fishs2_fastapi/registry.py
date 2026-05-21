from __future__ import annotations

import threading
from dataclasses import dataclass

from .errors import unknown_model
from .settings import settings


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip().lower()


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    backend_model_id: str

    def to_openai(self):
        from .api_models import OpenAIModel

        return OpenAIModel(id=self.model_id, owned_by="fishs2-fastapi")


class ModelRegistry:
    def __init__(self):
        self._models: list[ModelInfo] = []
        self._lookup: dict[str, ModelInfo] = {}
        self._lock = threading.RLock()

    def discover(self) -> list[ModelInfo]:
        default_model = settings.default_model.strip() or "fishaudio/s2-pro"
        raw_ids = [default_model, *settings.model_aliases]

        discovered: list[ModelInfo] = []
        lookup: dict[str, ModelInfo] = {}
        seen: set[str] = set()

        for model_id in raw_ids:
            normalized_id = _normalize_model_id(model_id)
            if not normalized_id or normalized_id in seen:
                continue

            info = ModelInfo(model_id=model_id, backend_model_id=default_model)
            discovered.append(info)
            lookup[normalized_id] = info
            seen.add(normalized_id)

        with self._lock:
            self._models = discovered
            self._lookup = lookup

        return discovered

    def refresh(self) -> list[ModelInfo]:
        return self.discover()

    def get(self, model_id: str) -> ModelInfo | None:
        normalized_id = _normalize_model_id(model_id)
        with self._lock:
            return self._lookup.get(normalized_id)

    def get_or_raise(self, model_id: str) -> ModelInfo:
        info = self.get(model_id)
        if info is None:
            raise unknown_model(model_id)
        return info

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            return list(self._models)

    def start_watching(self) -> bool:
        return False

    def stop_watching(self) -> None:
        return None


registry = ModelRegistry()
