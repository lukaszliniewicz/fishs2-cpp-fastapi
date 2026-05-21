from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .settings import settings

logger = logging.getLogger(__name__)

S2_SUCCESS_CODE = 1

S2_WARNING_CODES: dict[int, str] = {
    -1: "Reference audio encode failed; synthesis continued without reference voice.",
    -2: "Reference audio could not be loaded; synthesis continued without reference voice.",
}

S2_ERROR_CODES: dict[int, str] = {
    0: "Pipeline is not initialized.",
    -3: "Failed to initialize KV cache.",
    -4: "Generation produced no frames.",
    -5: "Audio decode failed.",
    -6: "Saving synthesized audio failed.",
    -7: "Reference audio is missing transcript text.",
    -8: "Reference prompt token count is zero.",
}

BACKEND_IDS = {
    "cpu": -1,
    "vulkan": 0,
    "cuda": 1,
    "metal": 2,
}

_DLL_BOOTSTRAP_LOCK = RLock()
_DLL_DIR_HANDLES: list[Any] = []
_ADDED_DLL_DIRS: set[str] = set()


class S2RuntimeError(RuntimeError):
    pass


class S2RuntimeUnavailable(S2RuntimeError):
    pass


class S2SynthesisError(S2RuntimeError):
    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)


@dataclass(slots=True)
class S2GenerateParams:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_tokens_before_end: int
    n_threads: int
    verbose: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _resolve_artifact_path(path: Path, *, label: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise S2RuntimeUnavailable(
            f"{label} not found at '{resolved}'. "
            f"Set FISHS2_{label.upper().replace(' ', '_')}_PATH to override."
        )
    return resolved


def resolve_s2_dll_path() -> Path:
    if settings.s2_dll_path is not None:
        explicit = _resolve_path(settings.s2_dll_path)
        if explicit.is_file():
            return explicit
        raise S2RuntimeUnavailable(
            f"Configured s2.dll path does not exist: '{explicit}'. "
            "Set FISHS2_S2_DLL_PATH to a valid file."
        )

    runtime_dir = _resolve_path(settings.runtime_dir)
    if runtime_dir.is_file() and runtime_dir.name.lower() == "s2.dll":
        return runtime_dir

    candidate = runtime_dir / "s2.dll"
    if candidate.is_file():
        return candidate

    raise S2RuntimeUnavailable(
        "s2.dll not found. Extract FishS2Sharp runtime files and set "
        "FISHS2_RUNTIME_DIR or FISHS2_S2_DLL_PATH."
    )


def ensure_nvidia_gpu_available() -> None:
    if settings.skip_gpu_check or not settings.require_nvidia_gpu:
        return

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise S2RuntimeUnavailable(
            "NVIDIA GPU was not detected. FishS2 server requires a CUDA-capable NVIDIA GPU."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        raise S2RuntimeUnavailable(
            "NVIDIA GPU was not detected. FishS2 server requires a CUDA-capable NVIDIA GPU. "
            f"Details: {detail}"
        )

    if platform.system() == "Windows":
        try:
            ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise S2RuntimeUnavailable(
                "NVIDIA driver runtime (nvcuda.dll) could not be loaded. "
                "Install/update NVIDIA drivers and retry."
            ) from exc


def _collect_dll_dirs(runtime_dir: Path) -> list[Path]:
    candidates: list[Path] = [runtime_dir]

    for extra in settings.runtime_extra_dll_dirs:
        candidates.append(_resolve_path(extra))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.append(prefix / "Library" / "bin")
        candidates.append(prefix / "Lib" / "site-packages" / "torch" / "lib")

    prefix = Path(sys.prefix)
    candidates.append(prefix / "Library" / "bin")
    candidates.append(prefix / "Lib" / "site-packages" / "torch" / "lib")

    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        logger.debug("Torch lib directory not available during DLL path bootstrap", exc_info=True)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)

    return unique


def bootstrap_dll_search_paths(runtime_dir: Path) -> None:
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    with _DLL_BOOTSTRAP_LOCK:
        for directory in _collect_dll_dirs(runtime_dir):
            if not directory.is_dir():
                continue

            key = str(directory).casefold()
            if key in _ADDED_DLL_DIRS:
                continue

            try:
                handle = os.add_dll_directory(str(directory))
            except OSError:
                logger.debug("Failed adding DLL search path", extra={"path": str(directory)}, exc_info=True)
                continue

            _DLL_DIR_HANDLES.append(handle)
            _ADDED_DLL_DIRS.add(key)


def _cstring(value: str) -> bytes:
    return value.encode("utf-8")


def _optional_cstring(value: str | None) -> bytes | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return text.encode("utf-8")


class _S2Native:
    def __init__(self, dll_path: Path):
        try:
            self.lib = ctypes.CDLL(str(dll_path))
        except OSError as exc:
            raise S2RuntimeUnavailable(
                "Failed loading s2.dll or one of its dependencies. "
                "Ensure FishS2Sharp runtime DLLs and CUDA runtime libraries are available. "
                f"Details: {exc}"
            ) from exc

        self.alloc_pipeline = self._bind("AllocS2Pipeline", [], ctypes.c_void_p)
        self.release_pipeline = self._bind("ReleaseS2Pipeline", [ctypes.c_void_p], None)

        self.set_log_level = self._bind("SetS2LogLevel", [ctypes.c_int32], None)
        self.sync_tokenizer_config = self._bind(
            "SyncS2TokenizerConfigFromS2Model",
            [ctypes.c_void_p, ctypes.c_void_p],
            None,
        )

        self.initialize_pipeline = self._bind(
            "InitializeS2Pipeline",
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
            ctypes.c_int,
        )

        self.alloc_generate_params = self._bind("AllocS2GenerateParams", [], ctypes.c_void_p)
        self.release_generate_params = self._bind("ReleaseS2GenerateParams", [ctypes.c_void_p], None)
        self.initialize_generate_params = self._bind(
            "InitializeS2GenerateParams",
            [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int,
            ],
            ctypes.c_int,
        )

        self.alloc_model = self._bind("AllocS2Model", [], ctypes.c_void_p)
        self.release_model = self._bind("ReleaseS2Model", [ctypes.c_void_p], None)
        self.initialize_model = self._bind(
            "InitializeS2Model",
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32],
            ctypes.c_int,
        )
        self.initialize_model_with_gpu_layers = self._bind(
            "InitializeS2ModelWithGpuLayers",
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32],
            ctypes.c_int,
            required=False,
        )

        self.alloc_tokenizer = self._bind("AllocS2Tokenizer", [], ctypes.c_void_p)
        self.release_tokenizer = self._bind("ReleaseS2Tokenizer", [ctypes.c_void_p], None)
        self.initialize_tokenizer = self._bind(
            "InitializeS2Tokenizer",
            [ctypes.c_void_p, ctypes.c_char_p],
            ctypes.c_int,
        )

        self.alloc_audio_codec = self._bind("AllocS2AudioCodec", [], ctypes.c_void_p)
        self.release_audio_codec = self._bind("ReleaseS2AudioCodec", [ctypes.c_void_p], None)
        self.initialize_audio_codec = self._bind(
            "InitializeS2AudioCodec",
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32],
            ctypes.c_int,
        )

        self.initialize_audio_codec_model_shared = self._bind(
            "InitializeS2AudioCodecModelShared",
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32],
            ctypes.c_int,
            required=False,
        )

        self.alloc_audio_buffer = self._bind("AllocS2AudioBuffer", [ctypes.c_int32], ctypes.c_void_p)
        self.release_audio_buffer = self._bind("ReleaseS2AudioBuffer", [ctypes.c_void_p], None)

        self.synthesize = self._bind(
            "S2Synthesize",
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int32),
            ],
            ctypes.c_int,
        )

        self.set_log_level(0)

    def _bind(
        self,
        name: str,
        argtypes: list[Any],
        restype: Any,
        *,
        required: bool = True,
    ):
        fn = getattr(self.lib, name, None)
        if fn is None:
            if required:
                raise S2RuntimeUnavailable(f"s2.dll is missing required export '{name}'.")
            return None

        fn.argtypes = argtypes
        fn.restype = restype
        return fn


class S2Runtime:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.backend = settings.backend.strip().lower() or "cuda"
        self._native: _S2Native | None = None
        self._model_handle: Any = None
        self._tokenizer_handle: Any = None
        self._audio_codec_handle: Any = None
        self._pipeline_handle: Any = None
        self._generate_params_handle: Any = None
        self._audio_buffer_handle: Any = None
        self._load_lock = RLock()
        self._loaded = False

    def _backend_id(self) -> int:
        if self.backend not in BACKEND_IDS:
            raise S2RuntimeUnavailable(
                f"Unsupported backend '{self.backend}'. Use one of: {', '.join(sorted(BACKEND_IDS))}."
            )
        return BACKEND_IDS[self.backend]

    def _load(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            if self.backend == "cuda":
                ensure_nvidia_gpu_available()

            model_path = _resolve_artifact_path(settings.model_path, label="model")
            tokenizer_path = _resolve_artifact_path(settings.tokenizer_path, label="tokenizer")
            dll_path = resolve_s2_dll_path()
            runtime_dir = dll_path.parent

            bootstrap_dll_search_paths(runtime_dir)
            native = _S2Native(dll_path)

            backend_id = self._backend_id()
            gpu_device = int(settings.gpu_device)
            n_gpu_layers = int(settings.n_gpu_layers)

            self._native = native
            self._model_handle = native.alloc_model()
            self._audio_codec_handle = native.alloc_audio_codec()
            self._tokenizer_handle = native.alloc_tokenizer()
            self._pipeline_handle = native.alloc_pipeline()
            self._generate_params_handle = native.alloc_generate_params()
            self._audio_buffer_handle = native.alloc_audio_buffer(-1)

            required_handles = [
                self._model_handle,
                self._audio_codec_handle,
                self._tokenizer_handle,
                self._pipeline_handle,
                self._generate_params_handle,
                self._audio_buffer_handle,
            ]
            if not all(required_handles):
                self.close()
                raise S2RuntimeUnavailable("Failed allocating internal FishS2 runtime objects.")

            model_path_bytes = _cstring(str(model_path))
            tokenizer_path_bytes = _cstring(str(tokenizer_path))

            model_ok: bool
            codec_ok: bool

            if n_gpu_layers >= 0:
                logger.info("Applying configured n_gpu_layers", extra={"n_gpu_layers": n_gpu_layers})
                init_with_layers = native.initialize_model_with_gpu_layers
                if init_with_layers is not None:
                    model_ok = (
                        init_with_layers(
                            self._model_handle,
                            model_path_bytes,
                            gpu_device,
                            backend_id,
                            n_gpu_layers,
                        )
                        == 1
                    )
                else:
                    logger.warning(
                        "Runtime export InitializeS2ModelWithGpuLayers is unavailable; "
                        "falling back to default model initialization",
                        extra={"n_gpu_layers": n_gpu_layers},
                    )
                    model_ok = (
                        native.initialize_model(
                            self._model_handle,
                            model_path_bytes,
                            gpu_device,
                            backend_id,
                        )
                        == 1
                    )

                codec_ok = (
                    native.initialize_audio_codec(
                        self._audio_codec_handle,
                        model_path_bytes,
                        gpu_device,
                        backend_id,
                    )
                    == 1
                )
                init_ok = model_ok and codec_ok
            else:
                shared_init = native.initialize_audio_codec_model_shared
                if shared_init is not None:
                    init_ok = (
                        shared_init(
                            self._model_handle,
                            self._audio_codec_handle,
                            model_path_bytes,
                            gpu_device,
                            backend_id,
                        )
                        == 1
                    )
                else:
                    model_ok = (
                        native.initialize_model(
                            self._model_handle,
                            model_path_bytes,
                            gpu_device,
                            backend_id,
                        )
                        == 1
                    )
                    codec_ok = (
                        native.initialize_audio_codec(
                            self._audio_codec_handle,
                            model_path_bytes,
                            gpu_device,
                            backend_id,
                        )
                        == 1
                    )
                    init_ok = model_ok and codec_ok

            if not init_ok:
                self.close()
                raise S2RuntimeUnavailable(
                    "Failed to initialize FishS2 model/audio codec. "
                    "Verify model GGUF path and GPU backend compatibility."
                )

            if native.initialize_tokenizer(self._tokenizer_handle, tokenizer_path_bytes) != 1:
                self.close()
                raise S2RuntimeUnavailable(
                    f"Failed to initialize tokenizer from '{tokenizer_path}'."
                )

            native.sync_tokenizer_config(self._model_handle, self._tokenizer_handle)

            if (
                native.initialize_pipeline(
                    self._pipeline_handle,
                    self._tokenizer_handle,
                    self._model_handle,
                    self._audio_codec_handle,
                )
                != 1
            ):
                self.close()
                raise S2RuntimeUnavailable("Failed to initialize FishS2 pipeline.")

            self._apply_generate_params(
                S2GenerateParams(
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    top_k=settings.top_k,
                    min_tokens_before_end=settings.min_tokens_before_end,
                    n_threads=settings.n_threads,
                    verbose=settings.verbose,
                )
            )

            self._loaded = True

    def _apply_generate_params(self, params: S2GenerateParams) -> None:
        if self._native is None or self._generate_params_handle is None:
            raise S2RuntimeUnavailable("FishS2 runtime is not initialized.")

        result = self._native.initialize_generate_params(
            self._generate_params_handle,
            int(params.max_new_tokens),
            float(params.temperature),
            float(params.top_p),
            int(params.top_k),
            int(params.min_tokens_before_end),
            int(params.n_threads),
            1 if params.verbose else 0,
        )
        if result != 1:
            raise S2RuntimeUnavailable("Failed to configure FishS2 generation parameters.")

    def configure_generate_params(self, params: S2GenerateParams) -> None:
        self._load()
        self._apply_generate_params(params)

    def synthesize_to_wav_bytes(
        self,
        *,
        text: str,
        params: S2GenerateParams,
        reference_audio_path: str | None,
        reference_text: str | None,
    ) -> bytes:
        self._load()
        self.configure_generate_params(params)

        if self._native is None:
            raise S2RuntimeUnavailable("FishS2 runtime is not initialized.")

        temp_dir = _resolve_path(settings.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"fishs2_{uuid4().hex}.wav"
        while output_path.exists():
            output_path = temp_dir / f"fishs2_{uuid4().hex}.wav"

        output_length = ctypes.c_int32(0)
        code: int
        try:
            code = self._native.synthesize(
                self._pipeline_handle,
                self._generate_params_handle,
                self._audio_buffer_handle,
                None,
                None,
                _optional_cstring(reference_audio_path),
                _cstring(reference_text or ""),
                _cstring(text),
                _cstring(str(output_path)),
                ctypes.byref(output_length),
            )

            if code in S2_WARNING_CODES:
                logger.warning(
                    "FishS2 synthesis warning",
                    extra={"code": code, "detail": S2_WARNING_CODES[code]},
                )
            elif code != S2_SUCCESS_CODE:
                detail = S2_ERROR_CODES.get(code, "Unknown synthesis error.")
                raise S2SynthesisError(code, f"FishS2 synthesis failed ({code}): {detail}")

            if not output_path.is_file() or output_path.stat().st_size <= 44:
                raise S2SynthesisError(code, "FishS2 synthesis did not produce a valid WAV file.")

            return output_path.read_bytes()
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed cleaning temporary FishS2 output", exc_info=True)

    def close(self) -> None:
        with self._load_lock:
            native = self._native
            if native is None:
                return

            if self._audio_buffer_handle:
                native.release_audio_buffer(self._audio_buffer_handle)
                self._audio_buffer_handle = None
            if self._generate_params_handle:
                native.release_generate_params(self._generate_params_handle)
                self._generate_params_handle = None
            if self._pipeline_handle:
                native.release_pipeline(self._pipeline_handle)
                self._pipeline_handle = None
            if self._tokenizer_handle:
                native.release_tokenizer(self._tokenizer_handle)
                self._tokenizer_handle = None
            if self._audio_codec_handle:
                native.release_audio_codec(self._audio_codec_handle)
                self._audio_codec_handle = None
            if self._model_handle:
                native.release_model(self._model_handle)
                self._model_handle = None

            self._native = None
            self._loaded = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
