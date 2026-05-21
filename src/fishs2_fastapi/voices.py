from __future__ import annotations

import json
import logging
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from .settings import settings

if TYPE_CHECKING:
    from .api_models import Voice, VoiceCreateResponse

logger = logging.getLogger(__name__)

VOICE_ID_INVALID_CHARS = re.compile(r"[^a-z0-9_-]+")
VOICE_ID_MULTI_HYPHEN = re.compile(r"-{2,}")


def normalize_voice_id(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = ascii_value.strip().lower()
    normalized = VOICE_ID_INVALID_CHARS.sub("-", normalized)
    normalized = VOICE_ID_MULTI_HYPHEN.sub("-", normalized)
    return normalized.strip("-_")


class VoiceStore:
    def __init__(self):
        self._base_dir = Path(settings.voices_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _voice_path(self, voice_id: str) -> Path:
        return self._base_dir / voice_id

    def _meta_path(self, voice_id: str) -> Path:
        return self._voice_path(voice_id) / "meta.json"

    def _sanitize_audio_name(self, name: str, index: int) -> str:
        base_name = Path(name).name if name else ""
        stem = Path(base_name).stem or f"sample_{index + 1}"
        suffix = Path(base_name).suffix.lower() or ".wav"
        safe_stem = normalize_voice_id(stem) or f"sample_{index + 1}"
        return f"{safe_stem}{suffix}"

    def create(
        self,
        voice_id: str,
        files: list[tuple[str, bytes]],
        *,
        model: str | None = None,
        language: str | None = None,
        prompt_text: str | None = None,
    ) -> VoiceCreateResponse:
        vpath = self._voice_path(voice_id)
        if vpath.exists():
            shutil.rmtree(vpath)
        vpath.mkdir(parents=True, exist_ok=True)

        sample_count = 0
        file_list: list[dict[str, str | int]] = []
        for index, (name, data) in enumerate(files):
            if not data:
                continue
            safe_name = self._sanitize_audio_name(name, index)
            dest = vpath / safe_name
            dest.write_bytes(data)
            file_list.append({"filename": dest.name, "size": len(data)})
            sample_count += 1

        created = int(time.time())
        meta = {
            "voice_id": voice_id,
            "created": created,
            "model": model,
            "language": language,
            "prompt_text": prompt_text,
            "files": file_list,
        }
        self._meta_path(voice_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        from .api_models import VoiceCreateResponse

        return VoiceCreateResponse(
            id=voice_id,
            model=model,
            language=language,
            sample_count=sample_count,
            created=created,
        )

    def register_staged_voices(self) -> int:
        if not self._base_dir.is_dir():
            return 0

        registered = 0
        entries = sorted(self._base_dir.iterdir(), key=lambda item: item.name.lower())
        for entry in entries:
            if not entry.is_dir():
                continue

            voice_id = entry.name
            meta_path = self._meta_path(voice_id)
            if meta_path.is_file():
                continue

            audio_files = [
                file_path
                for file_path in sorted(entry.iterdir(), key=lambda item: item.name.lower())
                if file_path.is_file() and file_path.name.lower() != "meta.json"
            ]
            if not audio_files:
                continue

            created = int(min(file_path.stat().st_mtime for file_path in audio_files))
            file_list = [
                {
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                }
                for file_path in audio_files
            ]

            meta = {
                "voice_id": voice_id,
                "created": created,
                "model": None,
                "language": None,
                "prompt_text": None,
                "files": file_list,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            registered += 1
            logger.info("Registered staged voice '%s' with %d sample(s)", voice_id, len(file_list))

        return registered

    def get(self, voice_id: str) -> dict | None:
        mpath = self._meta_path(voice_id)
        if not mpath.is_file():
            return None
        return json.loads(mpath.read_text(encoding="utf-8"))

    def delete(self, voice_id: str) -> bool:
        vpath = self._voice_path(voice_id)
        if not vpath.exists():
            return False
        shutil.rmtree(vpath)
        return True

    def list_all(self) -> list[Voice]:
        from .api_models import Voice

        voices: list[Voice] = []
        if not self._base_dir.is_dir():
            return voices
        for entry in sorted(self._base_dir.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            voices.append(Voice(**meta))
        return voices

    def get_sample_paths(self, voice_id: str) -> list[Path]:
        vpath = self._voice_path(voice_id)
        if not vpath.is_dir():
            return []

        samples: list[Path] = []
        for file_path in sorted(vpath.iterdir(), key=lambda item: item.name.lower()):
            if not file_path.is_file():
                continue
            if file_path.name.lower() == "meta.json":
                continue
            samples.append(file_path)
        return samples

    def resolve_for_speech(
        self,
        voice_id: str,
        *,
        prompt_text_override: str | None = None,
    ) -> tuple[str | None, str | None]:
        meta = self.get(voice_id)
        if meta is None:
            return None, None

        sample_paths = self.get_sample_paths(voice_id)
        if not sample_paths:
            return None, None

        prompt_text = prompt_text_override if prompt_text_override is not None else meta.get("prompt_text")

        reference_wav_path = str(sample_paths[0])
        return reference_wav_path, prompt_text


voice_store = VoiceStore()
