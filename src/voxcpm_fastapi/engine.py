from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import TYPE_CHECKING

from .audio import numpy_to_wav
from .errors import APIError, invalid_reference_audio, reference_audio_too_short, unknown_voice
from .model_loader import VoxCPMWrapper
from .settings import settings
from .voices import voice_store

if TYPE_CHECKING:
    from .api_models import CreateSpeechRequest, VoxCPMParams
    from .registry import ModelInfo

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self):
        self._models: dict[str, VoxCPMWrapper] = {}
        self._locks: dict[str, Lock] = {}

    def _get_lock(self, model_id: str) -> Lock:
        if model_id not in self._locks:
            self._locks[model_id] = Lock()
        return self._locks[model_id]

    def _get_wrapper(self, backend_model_id: str) -> VoxCPMWrapper:
        if backend_model_id not in self._models:
            self._models[backend_model_id] = VoxCPMWrapper(backend_model_id)
        return self._models[backend_model_id]

    def _resolve_voice_paths(self, request: CreateSpeechRequest) -> tuple[str | None, str | None, str | None]:
        mode = request.mode
        prompt_text = request.prompt_text

        if request.speaker_wav:
            reference = request.speaker_wav[0]
            prompt_wav = reference if mode == "hifi" else None
            return reference, prompt_wav, prompt_text

        voice_id = (request.voice_id or "").strip()
        if not voice_id or voice_id.lower() in {"default", "random", "alloy"}:
            return None, None, prompt_text

        reference_wav, prompt_wav, stored_prompt = voice_store.resolve_for_speech(
            voice_id,
            mode_override=mode,
            prompt_text_override=prompt_text,
        )
        if reference_wav is None:
            raise unknown_voice(voice_id)

        if prompt_text is None:
            prompt_text = stored_prompt

        return reference_wav, prompt_wav, prompt_text

    def _validate_reference_audio_path(self, path: str | None) -> None:
        if not path:
            return

        min_seconds = settings.min_reference_audio_seconds
        if min_seconds <= 0:
            return

        import soundfile as sf

        try:
            info = sf.info(path)
        except Exception as exc:
            raise invalid_reference_audio(path) from exc

        duration = 0.0
        if info.samplerate and info.samplerate > 0 and info.frames is not None:
            duration = float(info.frames) / float(info.samplerate)

        if duration < min_seconds:
            raise reference_audio_too_short(path, duration, min_seconds)

    def _build_generate_kwargs(
        self,
        request: CreateSpeechRequest,
        *,
        reference_wav_path: str | None,
        prompt_wav_path: str | None,
        prompt_text: str | None,
    ) -> dict:
        kwargs = {
            "cfg_value": settings.cfg_value,
            "inference_timesteps": settings.inference_timesteps,
            "normalize": settings.normalize,
            "denoise": settings.denoise,
            "retry_badcase": settings.retry_badcase,
            "retry_badcase_max_times": settings.retry_badcase_max_times,
            "retry_badcase_ratio_threshold": settings.retry_badcase_ratio_threshold,
            "min_len": settings.min_len,
            "max_len": settings.max_len,
        }

        params: VoxCPMParams | None = request.voxcpm
        if params is not None:
            for field_name in (
                "cfg_value",
                "inference_timesteps",
                "normalize",
                "denoise",
                "retry_badcase",
                "retry_badcase_max_times",
                "retry_badcase_ratio_threshold",
                "min_len",
                "max_len",
            ):
                value = getattr(params, field_name)
                if value is not None:
                    kwargs[field_name] = value

        text = request.input
        control = (request.control or "").strip()
        if control and not (prompt_wav_path and prompt_text):
            text = f"({control}){text}"

        if request.speed != 1.0:
            logger.info("Ignoring unsupported speed override for VoxCPM", extra={"speed": request.speed})

        kwargs["text"] = text
        if reference_wav_path is not None:
            kwargs["reference_wav_path"] = reference_wav_path
        if prompt_wav_path is not None:
            kwargs["prompt_wav_path"] = prompt_wav_path
        if prompt_text is not None:
            kwargs["prompt_text"] = prompt_text

        return kwargs

    def generate_speech(self, request: CreateSpeechRequest, model_info: ModelInfo):
        backend_model_id = model_info.backend_model_id
        wrapper = self._get_wrapper(backend_model_id)
        lock = self._get_lock(backend_model_id)

        reference_wav_path, prompt_wav_path, prompt_text = self._resolve_voice_paths(request)
        if request.mode == "hifi" and reference_wav_path is None:
            raise APIError(
                "mode 'hifi' requires reference audio via voice or speaker_wav",
                param="mode",
                code="missing_reference_audio",
                status=422,
            )
        if request.mode == "hifi" and not prompt_text:
            raise APIError(
                "mode 'hifi' requires prompt_text",
                param="prompt_text",
                code="missing_prompt_text",
                status=422,
            )

        self._validate_reference_audio_path(reference_wav_path)
        self._validate_reference_audio_path(prompt_wav_path)

        kwargs = self._build_generate_kwargs(
            request,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )

        with lock:
            try:
                waveform = wrapper.generate(**kwargs)
            except FileNotFoundError as exc:
                raise APIError(str(exc), param="speaker_wav", code="missing_reference_audio", status=422) from exc
            except ValueError as exc:
                raise APIError(str(exc), code="invalid_generation_request", status=422) from exc

        return numpy_to_wav(waveform, wrapper.sample_rate)

    async def generate_speech_async(self, request: CreateSpeechRequest, model_info: ModelInfo):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_speech, request, model_info)

    def refresh(self):
        self._models.clear()


engine = InferenceEngine()
