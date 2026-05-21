from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "voxcpm-fastapi"


class ModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class VoiceFile(BaseModel):
    filename: str
    size: int


class Voice(BaseModel):
    voice_id: str
    object: str = "voice"
    files: list[VoiceFile]
    created: int
    model: str | None = None
    language: str | None = None
    mode: Literal["reference", "hifi"] = "reference"
    prompt_text: str | None = None


class VoiceList(BaseModel):
    object: str = "list"
    data: list[Voice]


class VoiceCreateResponse(BaseModel):
    id: str
    object: str = "voice"
    model: str | None = None
    language: str | None = None
    sample_count: int
    created: int
    mode: Literal["reference", "hifi"] = "reference"


class VoxCPMParams(BaseModel):
    cfg_value: float | None = Field(default=None, gt=0.0, le=20.0)
    inference_timesteps: int | None = Field(default=None, ge=1, le=200)
    normalize: bool | None = None
    denoise: bool | None = None
    retry_badcase: bool | None = None
    retry_badcase_max_times: int | None = Field(default=None, ge=1, le=20)
    retry_badcase_ratio_threshold: float | None = Field(default=None, gt=0.0, le=50.0)
    min_len: int | None = Field(default=None, ge=1)
    max_len: int | None = Field(default=None, ge=1)


class VoiceIdentifier(BaseModel):
    id: str


class CreateSpeechRequest(BaseModel):
    model: str
    input: str = Field(..., min_length=1, max_length=12000)
    voice: str | VoiceIdentifier = "default"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=4096)

    voxcpm: VoxCPMParams | None = None
    speaker_wav: list[str] | None = None
    mode: Literal["reference", "hifi"] | None = None
    prompt_text: str | None = Field(default=None, max_length=4096)
    control: str | None = Field(default=None, max_length=512)

    @property
    def voice_id(self) -> str:
        if isinstance(self.voice, VoiceIdentifier):
            return self.voice.id
        return self.voice
