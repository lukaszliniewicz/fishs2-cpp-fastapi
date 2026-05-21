from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "fishs2-fastapi"


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


class FishS2Params(BaseModel):
    max_new_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    min_tokens_before_end: int | None = Field(default=None, ge=0, le=8192)
    n_threads: int | None = Field(default=None, ge=0, le=512)
    verbose: bool | None = None


class VoiceIdentifier(BaseModel):
    id: str


class CreateSpeechRequest(BaseModel):
    model: str
    input: str = Field(..., min_length=1, max_length=12000)
    voice: str | VoiceIdentifier = "default"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=4096)

    fishs2: FishS2Params | None = None
    reference_audio: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reference_audio", "prompt_audio", "ref_audio"),
        description="Local reference audio path for voice cloning (Fish canonical naming)",
    )
    reference_text: str | None = Field(
        default=None,
        max_length=4096,
        validation_alias=AliasChoices("reference_text", "ref_text"),
        description="Transcript for reference audio (Fish canonical naming)",
    )

    # Compatibility with existing wrappers/clients.
    speaker_wav: list[str] | None = None
    prompt_text: str | None = Field(default=None, max_length=4096)
    control: str | None = Field(default=None, max_length=512)

    @property
    def voice_id(self) -> str:
        if isinstance(self.voice, VoiceIdentifier):
            return self.voice.id
        return self.voice
