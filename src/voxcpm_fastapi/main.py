from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api_models import CreateSpeechRequest, ModelList, VoxCPMParams, VoiceCreateResponse, VoiceList
from .audio import SUPPORTED_FORMATS
from .engine import engine
from .errors import APIError
from .logging_setup import (
    ACCESS_LOGGER_NAME,
    APP_LOGGER_NAME,
    ERROR_LOGGER_NAME,
    configure_file_logging,
    reset_request_id,
    set_request_id,
)
from .registry import registry
from .settings import settings
from .voices import normalize_voice_id, normalize_voice_mode, voice_store

configure_file_logging(
    logs_dir=settings.logs_dir,
    level=settings.log_level,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count,
    app_log_file=settings.app_log_file,
    access_log_file=settings.access_log_file,
    error_log_file=settings.error_log_file,
)

app_logger = logging.getLogger(APP_LOGGER_NAME)
access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
error_logger = logging.getLogger(ERROR_LOGGER_NAME)

INSTRUCTION_VOX_FIELDS = set(VoxCPMParams.model_fields.keys())

app = FastAPI(
    title="VoxCPM2 FastAPI Server",
    description="OpenAI-compatible text-to-speech wrapper for VoxCPM2",
    version=__version__,
    docs_url="/",
)


def _sanitize_request_id(raw_request_id: str) -> str:
    allowed = {"-", "_", "."}
    sanitized = "".join(ch for ch in raw_request_id.strip() if ch.isalnum() or ch in allowed)
    return sanitized[:128]


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    header_name = settings.request_id_header or "X-Request-ID"
    incoming_request_id = request.headers.get(header_name, "")
    request_id = _sanitize_request_id(incoming_request_id) or uuid4().hex

    token = set_request_id(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    response: Response | None = None

    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        status = response.status_code if response is not None else 500
        client_host = request.client.host if request.client is not None else "-"
        content_length = "-"
        if response is not None:
            content_length = response.headers.get("content-length", "-")
            response.headers.setdefault(header_name, request_id)

        access_logger.info(
            "request_complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": f"{duration_ms:.2f}",
                "client": client_host,
                "content_length": content_length,
            },
        )
        reset_request_id(token)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    record = {
        "method": request.method,
        "path": request.url.path,
        "status": exc.status,
        "code": exc.code or "invalid_request_error",
        "param": exc.param,
        "detail": exc.message,
    }

    if exc.status >= 500:
        error_logger.error("api_error", extra=record)
    else:
        app_logger.info("api_error", extra=record)

    return exc.to_response()


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    app_logger.info(
        "request_validation_error",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": 422,
            "detail": exc.errors(),
        },
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    details = {
        "method": request.method,
        "path": request.url.path,
        "status": exc.status_code,
        "detail": exc.detail,
    }
    if exc.status_code >= 500:
        error_logger.error("http_exception", extra=details)
    else:
        app_logger.info("http_exception", extra=details)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    error_logger.error(
        "unhandled_exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": 500,
            "detail": str(exc),
        },
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "param": None,
                "code": "internal_server_error",
            }
        },
        status_code=500,
    )


@app.on_event("startup")
async def startup():
    registered = voice_store.register_staged_voices()
    if registered:
        app_logger.info("Registered %d staged voice(s) from %s", registered, settings.voices_dir)
    registry.discover()


def _parse_instruction_overrides(instructions: str | None) -> dict:
    if instructions is None:
        return {}

    raw = instructions.strip()
    if not raw or not raw.startswith("{"):
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(
            "instructions must be valid JSON when used for VoxCPM overrides",
            param="instructions",
            code="invalid_instructions_json",
            status=422,
        ) from exc

    if not isinstance(payload, dict):
        raise APIError(
            "instructions JSON must decode to an object",
            param="instructions",
            code="invalid_instructions_json",
            status=422,
        )

    voxcpm_overrides: dict[str, object] = {}
    raw_voxcpm = payload.get("voxcpm")
    if raw_voxcpm is not None:
        if not isinstance(raw_voxcpm, dict):
            raise APIError(
                "instructions.voxcpm must be a JSON object",
                param="instructions",
                code="invalid_instructions_voxcpm",
                status=422,
            )
        voxcpm_overrides.update(raw_voxcpm)

    for key in INSTRUCTION_VOX_FIELDS:
        if key in payload:
            voxcpm_overrides[key] = payload[key]

    scalar_fields: dict[str, object] = {}
    for key in ("mode", "prompt_text", "control"):
        if key in payload:
            scalar_fields[key] = payload[key]

    mode = scalar_fields.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise APIError(
            "instructions.mode must be a string",
            param="instructions",
            code="invalid_instructions_mode",
            status=422,
        )

    prompt_text = scalar_fields.get("prompt_text")
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise APIError(
            "instructions.prompt_text must be a string",
            param="instructions",
            code="invalid_instructions_prompt_text",
            status=422,
        )

    control = scalar_fields.get("control")
    if control is not None and not isinstance(control, str):
        raise APIError(
            "instructions.control must be a string",
            param="instructions",
            code="invalid_instructions_control",
            status=422,
        )

    return {
        "voxcpm": voxcpm_overrides or None,
        **scalar_fields,
    }


def _apply_instruction_overrides(body: CreateSpeechRequest) -> CreateSpeechRequest:
    overrides = _parse_instruction_overrides(body.instructions)
    if not overrides:
        return body

    payload = body.model_dump()

    voxcpm_overrides = overrides.get("voxcpm")
    if voxcpm_overrides is not None:
        existing_voxcpm = {}
        if body.voxcpm is not None:
            existing_voxcpm = body.voxcpm.model_dump(exclude_none=True)
        merged_voxcpm = {**voxcpm_overrides, **existing_voxcpm}
        try:
            payload["voxcpm"] = VoxCPMParams.model_validate(merged_voxcpm)
        except ValidationError as exc:
            details = exc.errors(include_url=False)
            message = details[0]["msg"] if details else str(exc)
            raise APIError(
                f"Invalid VoxCPM overrides in instructions: {message}",
                param="instructions",
                code="invalid_instructions_voxcpm",
                status=422,
            ) from exc

    for key in ("mode", "prompt_text", "control"):
        value = overrides.get(key)
        if value is None:
            continue
        if payload.get(key) is None:
            payload[key] = value

    try:
        return CreateSpeechRequest.model_validate(payload)
    except ValidationError as exc:
        details = exc.errors(include_url=False)
        message = details[0]["msg"] if details else str(exc)
        raise APIError(
            f"Invalid overrides in instructions: {message}",
            param="instructions",
            code="invalid_instructions_payload",
            status=422,
        ) from exc


def _derive_voice_id(voice_id: str | None, name: str | None, first_filename: str | None) -> str:
    raw = (voice_id or "").strip()
    if not raw:
        raw = (name or "").strip()
    if not raw and first_filename:
        raw = Path(first_filename).stem

    normalized = normalize_voice_id(raw)
    if normalized:
        return normalized
    return f"voice-{int(time.time())}"


async def _read_uploads(
    *,
    files: list[UploadFile] | None = None,
    audio_sample: UploadFile | None = None,
    file: UploadFile | None = None,
) -> list[tuple[str, bytes]]:
    uploads: list[UploadFile] = []
    if files:
        uploads.extend(files)
    if audio_sample is not None:
        uploads.append(audio_sample)
    if file is not None:
        uploads.append(file)

    data: list[tuple[str, bytes]] = []
    for upload in uploads:
        payload = await upload.read()
        if not payload:
            continue
        filename = upload.filename or f"sample_{len(data) + 1}.wav"
        data.append((filename, payload))
    return data


def _validate_mode(mode: str | None) -> str:
    try:
        return normalize_voice_mode(mode)
    except ValueError as exc:
        raise APIError(str(exc), param="mode", code="invalid_mode", status=422) from exc


async def _create_voice_from_uploads(
    *,
    files: list[UploadFile] | None,
    audio_sample: UploadFile | None,
    file: UploadFile | None,
    voice_id: str | None,
    name: str | None,
    model: str | None,
    language: str | None,
    mode: str | None,
    prompt_text: str | None,
) -> VoiceCreateResponse:
    uploaded = await _read_uploads(files=files, audio_sample=audio_sample, file=file)
    if not uploaded:
        raise APIError("At least one audio file is required", param="files", code="missing_files")

    first_filename = uploaded[0][0]
    normalized_mode = _validate_mode(mode)
    resolved_voice_id = _derive_voice_id(voice_id, name, first_filename)

    return voice_store.create(
        resolved_voice_id,
        uploaded,
        model=model,
        language=language,
        mode=normalized_mode,
        prompt_text=prompt_text,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "model_count": len(registry.list_models()),
        "voice_count": len(voice_store.list_all()),
        "device": settings.device,
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    registry.discover()
    return ModelList(data=[model.to_openai() for model in registry.list_models()])


@app.get("/v1/audio/voices", response_model=VoiceList)
@app.get("/v1/voices", response_model=VoiceList)
@app.get("/v1/files", response_model=VoiceList)
async def list_voices():
    return VoiceList(data=voice_store.list_all())


@app.post("/v1/audio/voices", response_model=VoiceCreateResponse)
async def create_voice(
    files: list[UploadFile] | None = File(default=None, description="Audio sample files"),
    audio_sample: UploadFile | None = File(default=None, description="Single audio sample"),
    voice_id: str | None = Form(default=None, description="Custom voice ID"),
    name: str | None = Form(default=None, description="Optional display name"),
    model: str | None = Form(default=None, description="Associated model ID"),
    language: str | None = Form(default=None, description="Language code"),
    purpose: str | None = Form(default=None, description="OpenAI-style compatibility field"),
    prompt_text: str | None = Form(default=None, description="Prompt transcript for hifi cloning"),
    mode: str | None = Form(default=None, description="reference or hifi"),
):
    _ = purpose
    return await _create_voice_from_uploads(
        files=files,
        audio_sample=audio_sample,
        file=None,
        voice_id=voice_id,
        name=name,
        model=model,
        language=language,
        mode=mode,
        prompt_text=prompt_text,
    )


@app.post("/v1/files", response_model=VoiceCreateResponse)
async def create_file_legacy(
    file: UploadFile | None = File(default=None, description="Legacy single upload field"),
    files: list[UploadFile] | None = File(default=None, description="XTTS-compatible upload field"),
    audio_sample: UploadFile | None = File(default=None, description="Alternative single upload field"),
    voice_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    purpose: str | None = Form(default=None),
    prompt_text: str | None = Form(default=None),
    mode: str | None = Form(default=None),
):
    _ = purpose
    return await _create_voice_from_uploads(
        files=files,
        audio_sample=audio_sample,
        file=file,
        voice_id=voice_id,
        name=name,
        model=model,
        language=language,
        mode=mode,
        prompt_text=prompt_text,
    )


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    if voice_store.delete(voice_id):
        return {"deleted": True, "id": voice_id}
    raise APIError(f"Voice '{voice_id}' not found", param="voice_id", code="voice_not_found", status=404)


@app.post("/v1/audio/speech")
async def create_speech(body: CreateSpeechRequest):
    response_format = (body.response_format or "").strip().lower()
    if response_format not in SUPPORTED_FORMATS:
        raise APIError(
            f"Unsupported response_format: {body.response_format}. VoxCPM wrapper currently supports only 'wav'.",
            param="response_format",
            code="unsupported_format",
            status=422,
        )

    body = _apply_instruction_overrides(body)

    model_info = registry.get(body.model)
    if model_info is None:
        raise APIError(
            f"Model '{body.model}' not found",
            param="model",
            code="model_not_found",
            status=404,
        )

    try:
        wav_bytes = await engine.generate_speech_async(body, model_info)
    except ImportError as exc:
        raise APIError(
            "VoxCPM runtime is not installed. Run the bootstrapper to install dependencies.",
            code="backend_not_available",
            status=503,
        ) from exc

    return Response(content=wav_bytes, media_type="audio/wav")
