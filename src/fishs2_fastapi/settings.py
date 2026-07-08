from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8020

    models_dir: Path = Path("models")
    voices_dir: Path = Path("voices")
    logs_dir: Path = Path("logs")
    temp_dir: Path = Path(".tmp")

    runtime_dir: Path = Path("runtime/fishs2sharp")
    s2_dll_path: Path | None = None
    runtime_extra_dll_dirs: list[Path] = Field(default_factory=list)

    app_log_file: str = "app.log"
    access_log_file: str = "access.log"
    error_log_file: str = "errors.log"
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    log_backup_count: int = Field(default=5, ge=1)
    request_id_header: str = "X-Request-ID"

    backend: str = "cuda"
    gpu_device: int = 0
    n_gpu_layers: int = -1
    require_nvidia_gpu: bool = True
    skip_gpu_check: bool = False

    default_model: str = "fishaudio/s2-pro"
    model_aliases: list[str] = Field(default_factory=lambda: ["fishs2", "fish-s2", "s2-pro"])
    model_path: Path = Path("models/s2-pro-q8_0.gguf")
    tokenizer_path: Path = Path("models/tokenizer.json")

    max_new_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0, le=500)
    min_tokens_before_end: int = Field(default=0, ge=0, le=8192)
    n_threads: int = Field(default=0, ge=0, le=512)
    verbose: bool = True

    min_reference_audio_seconds: float = Field(default=0.0, ge=0.0)

    model_config = {"env_prefix": "fishs2_", "env_file": ".env", "extra": "ignore"}

    @field_validator("model_aliases", mode="before")
    @classmethod
    def _parse_model_aliases(cls, value: Any) -> list[str]:
        if value is None:
            return ["fishs2", "fish-s2", "s2-pro"]

        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]

        if isinstance(value, (list, tuple, set)):
            parsed: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    parsed.append(text)
            return parsed

        raise TypeError("model_aliases must be a list or comma-separated string")

    @field_validator("runtime_extra_dll_dirs", mode="before")
    @classmethod
    def _parse_runtime_dirs(cls, value: Any) -> list[Path]:
        if value is None:
            return []

        if isinstance(value, str):
            if not value.strip():
                return []
            raw_items = [item.strip() for item in value.replace(";", ",").split(",")]
            return [Path(item) for item in raw_items if item]

        if isinstance(value, (list, tuple, set)):
            parsed: list[Path] = []
            for item in value:
                text = str(item).strip()
                if text:
                    parsed.append(Path(text))
            return parsed

        raise TypeError("runtime_extra_dll_dirs must be a list or a delimited string")


settings = Settings()
