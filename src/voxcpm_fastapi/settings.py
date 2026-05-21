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

    app_log_file: str = "app.log"
    access_log_file: str = "access.log"
    error_log_file: str = "errors.log"
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    log_backup_count: int = Field(default=5, ge=1)
    request_id_header: str = "X-Request-ID"

    device: str = "auto"
    default_model: str = "openbmb/VoxCPM2"
    model_aliases: list[str] = Field(default_factory=lambda: ["voxcpm2"])
    optimize: bool = False
    load_denoiser: bool = False

    cfg_value: float = 1.5
    inference_timesteps: int = 15
    normalize: bool = False
    denoise: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = 3
    retry_badcase_ratio_threshold: float = 6.0
    min_len: int = 2
    max_len: int = 4096

    min_reference_audio_seconds: float = Field(default=0.0, ge=0.0)

    model_config = {"env_prefix": "voxcpm_", "env_file": ".env"}

    @field_validator("model_aliases", mode="before")
    @classmethod
    def _parse_model_aliases(cls, value: Any) -> list[str]:
        if value is None:
            return ["voxcpm2"]

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


settings = Settings()
