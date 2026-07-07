#!/usr/bin/env python3
"""FishS2 FastAPI bootstrapper: verify hardware, download assets, start server."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("run")

PROJECT_DIR = Path(__file__).parent.resolve()
SERVER_MODULE = "fishs2_fastapi.main:app"

DEFAULT_RUNTIME_ZIP_URL = (
    "https://github.com/subspecs/FishS2Sharp/releases/download/v1.0.2/"
    "FishS2Sharp-Win-x86_64-CUDA-VULKAN-CPU.zip"
)
DEFAULT_RUNTIME_ZIP_SHA256 = "15c2036ce0a3a5e6d4c8f81a2834e683f7f40a8155179c66858fd27a15d4cc74"
DEFAULT_HF_REPO = "rodrigomt/s2-pro-gguf"
DEFAULT_MODEL_QUANT = "q8_0"
MODEL_QUANT_FILENAMES = {
    "f16": "s2-pro-f16.gguf",
    "q8_0": "s2-pro-q8_0.gguf",
    "q6_k": "s2-pro-q6_k.gguf",
    "q5_k_m": "s2-pro-q5_k_m.gguf",
    "q4_k_m": "s2-pro-q4_k_m.gguf",
    "q3_k": "s2-pro-q3_k.gguf",
    "q2_k": "s2-pro-q2_k.gguf",
}
DEFAULT_MODEL_FILENAME = MODEL_QUANT_FILENAMES[DEFAULT_MODEL_QUANT]
DEFAULT_TOKENIZER_FILENAME = "tokenizer.json"

PIXI_PATH_OVERRIDE: Path | None = None


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def _resolve_env_path(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return _resolve_path(raw)
    return fallback


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.2f} GiB"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _ensure_local_cache_env() -> None:
    os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_DIR / ".pip-cache"))
    os.environ.setdefault("PIXI_CACHE_DIR", str(PROJECT_DIR / ".pixi-cache"))
    os.environ.setdefault("RATTLER_CACHE_DIR", str(PROJECT_DIR / ".pixi-cache" / "rattler"))
    os.environ.setdefault("TMP", str(PROJECT_DIR / ".tmp"))
    os.environ.setdefault("TEMP", str(PROJECT_DIR / ".tmp"))

    hf_home = PROJECT_DIR / ".hf"
    hf_hub_cache = hf_home / "hub"
    hf_assets_cache = hf_home / "assets"
    hf_datasets_cache = hf_home / "datasets"
    transformers_cache = hf_home / "transformers"

    for directory in (
        Path(os.environ["PIP_CACHE_DIR"]),
        Path(os.environ["PIXI_CACHE_DIR"]),
        Path(os.environ["RATTLER_CACHE_DIR"]),
        Path(os.environ["TMP"]),
        hf_home,
        hf_hub_cache,
        hf_assets_cache,
        hf_datasets_cache,
        transformers_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("HF_ASSETS_CACHE", str(hf_assets_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, *, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_target = destination.with_suffix(destination.suffix + ".part")
    if temp_target.exists():
        temp_target.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "fishs2-fastapi-bootstrap/0.1"})
    log.info("Downloading %s", label)
    log.info("  from: %s", url)
    log.info("  to:   %s", destination)

    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_target.open("wb") as output:
            total_header = response.headers.get("Content-Length")
            total_size = int(total_header) if total_header and total_header.isdigit() else 0
            downloaded = 0
            progress_step = 256 * 1024 * 1024
            next_report = progress_step

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)

                if total_size > 0 and downloaded >= next_report:
                    percent = min(100.0, (downloaded / total_size) * 100.0)
                    log.info(
                        "  progress: %.1f%% (%s / %s)",
                        percent,
                        _format_bytes(downloaded),
                        _format_bytes(total_size),
                    )
                    next_report += progress_step

            if total_size > 0 and downloaded != total_size:
                raise RuntimeError(
                    f"Download size mismatch for {label}: expected {total_size} bytes, got {downloaded} bytes"
                )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temp_target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading {label}: {exc}") from exc

    temp_target.replace(destination)


def _ensure_file(
    path: Path,
    *,
    url: str,
    expected_sha256: str | None,
    force: bool,
    label: str,
) -> None:
    expected = (expected_sha256 or "").strip().lower() or None

    if path.is_file() and not force:
        if expected is None:
            log.info("Using existing %s at %s", label, path)
            return

        actual = _sha256_file(path)
        if actual == expected:
            log.info("Using existing %s at %s (sha256 verified)", label, path)
            return

        log.warning("Existing %s hash mismatch; re-downloading", label)

    _download_file(url, path, label=label)

    if expected is not None:
        actual = _sha256_file(path)
        if actual != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {label}: expected {expected}, got {actual}. "
                "The downloaded file was removed."
            )


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(destination)


def _hf_resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"


def _resolve_model_quant(raw_value: str | None) -> str:
    quant = (raw_value or DEFAULT_MODEL_QUANT).strip().lower()
    if quant not in MODEL_QUANT_FILENAMES:
        allowed = ", ".join(sorted(MODEL_QUANT_FILENAMES))
        raise RuntimeError(f"Unsupported model quant '{quant}'. Allowed values: {allowed}")
    return quant


def _compile_runtime_from_source(backend: str, runtime_dir: Path) -> None:
    log.info("libs2 dynamic library is missing. Starting automated compilation from source...")

    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required build tool '{tool}' is not installed or not in PATH.")

    build_root = PROJECT_DIR / ".tmp" / "s2_build"
    if build_root.exists():
        try:
            shutil.rmtree(build_root)
        except Exception:
            pass
    build_root.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Cloning rodrigomatta/s2.cpp recursively...")
        subprocess.run(
            ["git", "clone", "--recursive", "https://github.com/rodrigomatta/s2.cpp", "s2.cpp"],
            cwd=build_root,
            check=True,
        )

        s2_dir = build_root / "s2.cpp"
        build_dir = s2_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_flags = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DS2_BUILD_SHARED_LIBRARIES=ON"]
        if backend == "vulkan":
            cmake_flags.extend(["-DS2_CUDA=OFF", "-DS2_VULKAN=ON"])
        elif backend == "cuda":
            cmake_flags.extend(["-DS2_CUDA=ON", "-DS2_VULKAN=OFF"])
        else:
            cmake_flags.extend(["-DS2_CUDA=OFF", "-DS2_VULKAN=OFF"])

        log.info("Configuring CMake: %s", " ".join(cmake_flags))
        subprocess.run(cmake_flags, cwd=build_dir, check=True)

        import multiprocessing
        cores = multiprocessing.cpu_count()
        build_cmd = ["cmake", "--build", ".", "--config", "Release", "-j", str(cores)]
        log.info("Compiling project: %s", " ".join(build_cmd))
        subprocess.run(build_cmd, cwd=build_dir, check=True)

        runtime_dir.mkdir(parents=True, exist_ok=True)
        copied = False
        lib_ext = "so" if platform.system() != "Darwin" else "dylib"

        for pattern in (f"libs2.{lib_ext}", f"libggml*.{lib_ext}"):
            for file_path in build_dir.rglob(pattern):
                if file_path.is_file() and not file_path.is_symlink():
                    dest = runtime_dir / file_path.name
                    log.info("Copying %s -> %s", file_path.name, dest)
                    shutil.copy2(file_path, dest)
                    copied = True

        target_lib = runtime_dir / f"libs2.{lib_ext}"
        if not copied or not target_lib.is_file():
            raise RuntimeError(f"Build completed, but {target_lib.name} was not found in build outputs.")

        log.info("Successfully compiled and installed s2.cpp runtime at %s", runtime_dir)

    finally:
        try:
            shutil.rmtree(build_root, ignore_errors=True)
        except Exception:
            pass


def _ensure_runtime_bundle(*, force: bool) -> None:
    runtime_dir = _resolve_env_path("FISHS2_RUNTIME_DIR", PROJECT_DIR / "runtime" / "fishs2sharp")
    os.environ["FISHS2_RUNTIME_DIR"] = str(runtime_dir)

    is_windows = platform.system() == "Windows"
    lib_name = "s2.dll" if is_windows else ("libs2.dylib" if platform.system() == "Darwin" else "libs2.so")

    if is_windows:
        required_files = [
            runtime_dir / "s2.dll",
            runtime_dir / "ggml-base.dll",
            runtime_dir / "ggml.dll",
            runtime_dir / "ggml-cpu.dll",
            runtime_dir / "ggml-cuda.dll",
            runtime_dir / "FishS2Sharp.dll",
        ]
        if not force and all(path.is_file() for path in required_files):
            log.info("FishS2 runtime already present at %s", runtime_dir)
            return
    else:
        if not force and (runtime_dir / lib_name).is_file():
            log.info("FishS2 runtime already present at %s", runtime_dir)
            return

    if not is_windows:
        backend = os.environ.get("FISHS2_BACKEND", "cuda").strip().lower()
        _compile_runtime_from_source(backend, runtime_dir)
        return

    runtime_zip_path = _resolve_env_path(
        "FISHS2_RUNTIME_ZIP_PATH",
        PROJECT_DIR / ".tmp" / "fishs2sharp_runtime.zip",
    )
    runtime_zip_url = (os.environ.get("FISHS2_RUNTIME_ZIP_URL") or DEFAULT_RUNTIME_ZIP_URL).strip()
    runtime_zip_sha = (os.environ.get("FISHS2_RUNTIME_ZIP_SHA256") or DEFAULT_RUNTIME_ZIP_SHA256).strip()

    _ensure_file(
        runtime_zip_path,
        url=runtime_zip_url,
        expected_sha256=runtime_zip_sha,
        force=force,
        label="FishS2 runtime bundle",
    )
    _safe_extract_zip(runtime_zip_path, runtime_dir)

    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            "FishS2 runtime extraction completed, but required files are missing: "
            + ", ".join(missing)
        )

    log.info("FishS2 runtime is ready at %s", runtime_dir)



def _ensure_model_artifacts(*, force: bool) -> None:
    repo_id = (os.environ.get("FISHS2_HF_REPO_ID") or DEFAULT_HF_REPO).strip()
    selected_quant = _resolve_model_quant(os.environ.get("FISHS2_MODEL_QUANT"))
    default_model_filename = MODEL_QUANT_FILENAMES[selected_quant]
    model_path = _resolve_env_path("FISHS2_MODEL_PATH", PROJECT_DIR / "models" / default_model_filename)
    tokenizer_path = _resolve_env_path("FISHS2_TOKENIZER_PATH", PROJECT_DIR / "models" / DEFAULT_TOKENIZER_FILENAME)

    if os.environ.get("FISHS2_MODEL_PATH"):
        log.info("Using explicit model path from FISHS2_MODEL_PATH: %s", model_path)
    else:
        log.info("Selected FishS2 model quant: %s (%s)", selected_quant, model_path.name)

    os.environ["FISHS2_MODEL_PATH"] = str(model_path)
    os.environ["FISHS2_TOKENIZER_PATH"] = str(tokenizer_path)

    model_url = (os.environ.get("FISHS2_MODEL_URL") or _hf_resolve_url(repo_id, model_path.name)).strip()
    tokenizer_url = (os.environ.get("FISHS2_TOKENIZER_URL") or _hf_resolve_url(repo_id, tokenizer_path.name)).strip()
    model_sha = (os.environ.get("FISHS2_MODEL_SHA256") or "").strip()
    tokenizer_sha = (os.environ.get("FISHS2_TOKENIZER_SHA256") or "").strip()

    _ensure_file(
        model_path,
        url=model_url,
        expected_sha256=model_sha,
        force=force,
        label=f"FishS2 model ({model_path.name})",
    )
    _ensure_file(
        tokenizer_path,
        url=tokenizer_url,
        expected_sha256=tokenizer_sha,
        force=force,
        label=f"FishS2 tokenizer ({tokenizer_path.name})",
    )

    log.info("FishS2 model artifacts are ready in %s", model_path.parent)


def _ensure_artifacts_or_exit(*, skip_downloads: bool, force_downloads: bool) -> None:
    if skip_downloads:
        log.info("Skipping artifact downloads (--skip-downloads)")
        return

    try:
        _ensure_runtime_bundle(force=force_downloads)
        _ensure_model_artifacts(force=force_downloads)
    except RuntimeError as exc:
        log.error("Artifact preparation failed: %s", exc)
        sys.exit(1)


def _find_pixi() -> str:
    if PIXI_PATH_OVERRIDE is not None:
        path = PIXI_PATH_OVERRIDE
    else:
        exe = "pixi.exe" if platform.system() == "Windows" else "pixi"
        path = PROJECT_DIR / "bin" / exe

    if not path.is_file():
        log.error("pixi not found at %s", path)
        if PIXI_PATH_OVERRIDE is not None:
            log.error("The value passed to --pixi-path must point to an existing pixi binary.")
        else:
            log.error("Run run.bat (Windows) or run.sh (Linux/macOS) first.")
        sys.exit(1)

    return str(path)


def detect_nvidia_gpu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    detail = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, detail


def ensure_cuda_requirements_or_exit(*, skip_gpu_check: bool, backend: str) -> None:
    if skip_gpu_check:
        log.warning("Skipping NVIDIA GPU detection (--skip-gpu-check)")
        os.environ["FISHS2_SKIP_GPU_CHECK"] = "true"
        return

    if backend != "cuda":
        return

    ok, detail = detect_nvidia_gpu()
    if ok:
        log.info("Detected NVIDIA GPU")
        if detail:
            first_line = detail.splitlines()[0]
            log.info("nvidia-smi: %s", first_line)
        return

    log.error("NVIDIA GPU not detected. FishS2 server requires CUDA for practical inference.")
    if detail:
        log.error("Detection details: %s", detail)
    sys.exit(1)


def start_server(*, host: str, port: int) -> None:
    pixi = _find_pixi()
    manifest = PROJECT_DIR / "pyproject.toml"

    env = os.environ.copy()
    if platform.system() != "Windows":
        runtime_dir = _resolve_env_path("FISHS2_RUNTIME_DIR", PROJECT_DIR / "runtime" / "fishs2sharp")
        ld_path = env.get("LD_LIBRARY_PATH", "")
        if ld_path:
            env["LD_LIBRARY_PATH"] = f"{runtime_dir}:{ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = str(runtime_dir)

    cmd = [
        pixi,
        "run",
        "--manifest-path",
        str(manifest),
        "python",
        "-m",
        "uvicorn",
        SERVER_MODULE,
        "--host",
        host,
        "--port",
        str(port),
        "--no-access-log",
    ]
    log.info("Starting FishS2 FastAPI server...\n")
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)



def main() -> None:
    _ensure_local_cache_env()

    parser = argparse.ArgumentParser(
        description="FishS2 FastAPI server bootstrapper",
    )
    parser.add_argument(
        "--pixi-path",
        help="Path to an existing pixi binary (used instead of project-local bin/pixi)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("FISHS2_HOST", "0.0.0.0"),
        help="Host interface for uvicorn (default: FISHS2_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FISHS2_PORT", "8020")),
        help="Port for uvicorn (default: FISHS2_PORT or 8020)",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "vulkan", "cpu"],
        default=os.environ.get("FISHS2_BACKEND", "cuda"),
        help="Backend hint for FishS2 runtime (default: cuda)",
    )
    parser.add_argument(
        "--model-quant",
        default=os.environ.get("FISHS2_MODEL_QUANT", DEFAULT_MODEL_QUANT),
        help=(
            "Model quant variant to download "
            f"(default: {DEFAULT_MODEL_QUANT}; options: {', '.join(sorted(MODEL_QUANT_FILENAMES))})"
        ),
    )
    parser.add_argument(
        "--model-q4",
        action="store_true",
        help="Shortcut for --model-quant q4_k_m",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=int(os.environ.get("FISHS2_N_GPU_LAYERS", "-1")),
        help=(
            "Number of transformer layers to offload to GPU "
            "(default: FISHS2_N_GPU_LAYERS or -1 for runtime default behavior)"
        ),
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip NVIDIA GPU detection before startup",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Do not auto-download runtime/model artifacts",
    )
    parser.add_argument(
        "--force-downloads",
        action="store_true",
        help="Force re-download of runtime/model artifacts",
    )
    args = parser.parse_args()

    global PIXI_PATH_OVERRIDE
    if args.pixi_path:
        PIXI_PATH_OVERRIDE = Path(args.pixi_path).expanduser().resolve()
        log.info("Using pixi binary: %s", PIXI_PATH_OVERRIDE)

    skip_downloads = args.skip_downloads or _bool_env("FISHS2_SKIP_DOWNLOADS", default=False)
    force_downloads = args.force_downloads or _bool_env("FISHS2_FORCE_DOWNLOADS", default=False)
    requested_quant = "q4_k_m" if args.model_q4 else args.model_quant

    try:
        selected_quant = _resolve_model_quant(requested_quant)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    os.environ["FISHS2_BACKEND"] = args.backend
    os.environ["FISHS2_MODEL_QUANT"] = selected_quant
    os.environ["FISHS2_N_GPU_LAYERS"] = str(args.n_gpu_layers)
    ensure_cuda_requirements_or_exit(skip_gpu_check=args.skip_gpu_check, backend=args.backend)
    _ensure_artifacts_or_exit(skip_downloads=skip_downloads, force_downloads=force_downloads)
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
