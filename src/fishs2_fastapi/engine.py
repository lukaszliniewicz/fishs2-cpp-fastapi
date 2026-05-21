from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import TYPE_CHECKING

from .errors import APIError, invalid_reference_audio, reference_audio_too_short, unknown_voice
from .model_loader import S2GenerateParams, S2Runtime, S2SynthesisError
from .settings import settings
from .voices import voice_store

if TYPE_CHECKING:
    from .api_models import CreateSpeechRequest, FishS2Params
    from .registry import ModelInfo

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self):
        self._runtimes: dict[str, S2Runtime] = {}
        self._locks: dict[str, Lock] = {}

    def _get_lock(self, model_id: str) -> Lock:
        if model_id not in self._locks:
            self._locks[model_id] = Lock()
        return self._locks[model_id]

    def _get_runtime(self, backend_model_id: str) -> S2Runtime:
        if backend_model_id not in self._runtimes:
            self._runtimes[backend_model_id] = S2Runtime(backend_model_id)
        return self._runtimes[backend_model_id]

    def _resolve_voice_inputs(self, request: CreateSpeechRequest) -> tuple[str | None, str | None]:
        prompt_text = request.reference_text or request.prompt_text

        explicit_reference = (request.reference_audio or "").strip()
        if explicit_reference:
            return explicit_reference, prompt_text

        if request.speaker_wav:
            reference = request.speaker_wav[0]
            return reference, prompt_text

        voice_id = (request.voice_id or "").strip()
        if not voice_id or voice_id.lower() in {"default", "random", "alloy"}:
            return None, prompt_text

        reference_wav, stored_prompt = voice_store.resolve_for_speech(
            voice_id,
            prompt_text_override=prompt_text,
        )
        if reference_wav is None:
            raise unknown_voice(voice_id)

        if prompt_text is None:
            prompt_text = stored_prompt

        return reference_wav, prompt_text

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

    def _build_generate_params(self, request: CreateSpeechRequest) -> S2GenerateParams:
        params = S2GenerateParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            min_tokens_before_end=settings.min_tokens_before_end,
            n_threads=settings.n_threads,
            verbose=settings.verbose,
        )

        overrides: FishS2Params | None = request.fishs2
        if overrides is None:
            return params

        if overrides.max_new_tokens is not None:
            params.max_new_tokens = overrides.max_new_tokens
        if overrides.temperature is not None:
            params.temperature = overrides.temperature
        if overrides.top_p is not None:
            params.top_p = overrides.top_p
        if overrides.top_k is not None:
            params.top_k = overrides.top_k
        if overrides.min_tokens_before_end is not None:
            params.min_tokens_before_end = overrides.min_tokens_before_end
        if overrides.n_threads is not None:
            params.n_threads = overrides.n_threads
        if overrides.verbose is not None:
            params.verbose = overrides.verbose

        return params

    def _build_text(self, request: CreateSpeechRequest) -> str:
        text = request.input
        control = (request.control or "").strip()
        if control:
            text = f"({control}){text}"

        if request.speed != 1.0:
            raise APIError(
                "FishS2 backend does not support speed override; use speed=1.0",
                param="speed",
                code="unsupported_speed",
                status=422,
            )

        return text

    def generate_speech(self, request: CreateSpeechRequest, model_info: ModelInfo) -> bytes:
        runtime = self._get_runtime(model_info.backend_model_id)
        lock = self._get_lock(model_info.backend_model_id)

        reference_audio_path, prompt_text = self._resolve_voice_inputs(request)
        if reference_audio_path is not None and not (prompt_text or "").strip():
            raise APIError(
                "Reference audio requires transcript via reference_text (or prompt_text)",
                param="reference_text",
                code="missing_prompt_text",
                status=422,
            )

        self._validate_reference_audio_path(reference_audio_path)
        params = self._build_generate_params(request)
        text = self._build_text(request)

        with lock:
            try:
                return runtime.synthesize_to_wav_bytes(
                    text=text,
                    params=params,
                    reference_audio_path=reference_audio_path,
                    reference_text=prompt_text,
                )
            except S2SynthesisError as exc:
                if exc.code == -7:
                    raise APIError(
                        "Reference audio requires transcript via reference_text (or prompt_text)",
                        param="reference_text",
                        code="missing_prompt_text",
                        status=422,
                    ) from exc
                if exc.code == -2 and reference_audio_path is not None:
                    raise invalid_reference_audio(reference_audio_path) from exc
                raise APIError(str(exc), code="generation_failed", status=500) from exc

    async def generate_speech_async(self, request: CreateSpeechRequest, model_info: ModelInfo) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_speech, request, model_info)

    def refresh(self) -> None:
        for runtime in self._runtimes.values():
            runtime.close()
        self._runtimes.clear()


engine = InferenceEngine()
