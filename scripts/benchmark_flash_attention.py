#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import site
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / ".tmp"
REF_WAV = ROOT / "voices" / "sample_male" / "sample_male_new.wav"
PROMPT_TEXT_FILE = ROOT / "prompts" / "sample_male.txt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class BenchResult:
    label: str
    times: list[float]
    mean_excluding_first: float


def _pip_env() -> dict[str, str]:
    env = os.environ.copy()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    env["TMP"] = str(TMP_DIR)
    env["TEMP"] = str(TMP_DIR)
    return env


def _run_pip(*args: str, check: bool = True) -> int:
    cmd = [sys.executable, "-m", "pip", *args]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=_pip_env(), check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"pip command failed: {' '.join(cmd)}")
    return result.returncode


def _cleanup_stale_flash_artifacts() -> None:
    site_packages = Path(site.getsitepackages()[0])
    stale_paths = list(site_packages.glob("~lash_attn*"))
    for stale in stale_paths:
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        elif stale.exists():
            stale.unlink(missing_ok=True)


def _resolve_flash_wheel_url() -> str:
    import run

    wheel_url, reason = run._resolve_flash_attn_wheel()
    if wheel_url is None:
        raise RuntimeError(f"No flash-attn wheel available: {reason}")
    print(reason)
    return wheel_url


def ensure_flash_attn_installed() -> None:
    wheel_url = _resolve_flash_wheel_url()
    _run_pip("install", "--no-deps", "--force-reinstall", wheel_url)
    if importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("flash_attn install command succeeded but import spec is missing")


def ensure_flash_attn_uninstalled() -> None:
    _run_pip("uninstall", "-y", "flash-attn", "flash_attn", check=False)
    _cleanup_stale_flash_artifacts()
    if importlib.util.find_spec("flash_attn") is not None:
        raise RuntimeError("flash_attn still importable after uninstall")


def _http_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _speech_request(base_url: str, payload: dict, timeout: int = 1800) -> int:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return len(data)


def _start_server(port: int) -> tuple[subprocess.Popen[bytes], Path, Path]:
    stdout_log = TMP_DIR / f"bench_uvicorn_{port}_stdout.log"
    stderr_log = TMP_DIR / f"bench_uvicorn_{port}_stderr.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)

    stdout_handle = stdout_log.open("wb")
    stderr_handle = stderr_log.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "voxcpm_fastapi.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=ROOT,
        env=_pip_env(),
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    return proc, stdout_log, stderr_log


def _wait_for_health(proc: subprocess.Popen[bytes], base_url: str, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            health = _http_json(f"{base_url}/health", timeout=5)
            if health.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Timed out waiting for server health")


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_condition(label: str, base_url: str, payload: dict, runs: int, discard: int) -> BenchResult:
    times: list[float] = []
    print(f"\n=== {label.upper()} ({runs} runs) ===")
    for idx in range(1, runs + 1):
        started = time.perf_counter()
        size = _speech_request(base_url, payload)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"run {idx}: {elapsed:.3f}s ({size} bytes)")

    effective = times[discard:]
    if not effective:
        raise ValueError("No samples left after discarding warmup runs")
    avg = statistics.mean(effective)
    print(f"avg (discard first {discard}): {avg:.3f}s")
    return BenchResult(label=label, times=times, mean_excluding_first=avg)


def build_payload(text: str) -> dict:
    if not REF_WAV.is_file():
        raise FileNotFoundError(f"Missing reference wav: {REF_WAV}")
    if not PROMPT_TEXT_FILE.is_file():
        raise FileNotFoundError(f"Missing prompt text file: {PROMPT_TEXT_FILE}")

    prompt_text = PROMPT_TEXT_FILE.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError("Prompt transcript is empty")

    return {
        "model": "voxcpm2",
        "input": text,
        "voice": "default",
        "mode": "hifi",
        "speaker_wav": [str(REF_WAV)],
        "prompt_text": prompt_text,
        "response_format": "wav",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark with and without flash-attn")
    parser.add_argument("--runs", type=int, default=5, help="Total runs per condition")
    parser.add_argument("--discard", type=int, default=1, help="Number of warmup runs to discard")
    parser.add_argument("--port", type=int, default=8030, help="Benchmark server port")
    parser.add_argument(
        "--text",
        default=(
            "Inspector, no one climbed through that second-floor window because the latch and sill are untouched. "
            "The intruder entered by the corridor door, and someone inside helped them after midnight."
        ),
        help="Target generation text",
    )
    parser.add_argument(
        "--keep-no-flash",
        action="store_true",
        help="Do not restore flash-attn after benchmark completes",
    )
    args = parser.parse_args()

    if args.runs <= args.discard:
        raise ValueError("--runs must be greater than --discard")

    payload = build_payload(args.text)
    base_url = f"http://127.0.0.1:{args.port}"
    had_flash_before = importlib.util.find_spec("flash_attn") is not None

    with_result: BenchResult | None = None
    without_result: BenchResult | None = None

    proc: subprocess.Popen[bytes] | None = None
    try:
        ensure_flash_attn_installed()
        proc, _, _ = _start_server(args.port)
        _wait_for_health(proc, base_url)
        with_result = run_condition("with_flash_attn", base_url, payload, args.runs, args.discard)
        _stop_server(proc)
        proc = None

        ensure_flash_attn_uninstalled()
        proc, _, _ = _start_server(args.port)
        _wait_for_health(proc, base_url)
        without_result = run_condition("without_flash_attn", base_url, payload, args.runs, args.discard)
    finally:
        if proc is not None:
            _stop_server(proc)

        if not args.keep_no_flash and had_flash_before:
            try:
                ensure_flash_attn_installed()
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: failed to restore flash-attn: {exc}")

    if with_result is None or without_result is None:
        raise RuntimeError("Benchmark did not complete")

    delta = with_result.mean_excluding_first - without_result.mean_excluding_first
    ratio = with_result.mean_excluding_first / without_result.mean_excluding_first

    print("\n=== SUMMARY ===")
    print(f"with_flash_attn avg:    {with_result.mean_excluding_first:.3f}s")
    print(f"without_flash_attn avg: {without_result.mean_excluding_first:.3f}s")
    print(f"delta (with - without): {delta:.3f}s")
    print(f"ratio (with/without):   {ratio:.3f}")

    payload_json = {
        "runs": args.runs,
        "discard": args.discard,
        "with_flash_attn": {
            "times": with_result.times,
            "avg_excluding_discard": with_result.mean_excluding_first,
        },
        "without_flash_attn": {
            "times": without_result.times,
            "avg_excluding_discard": without_result.mean_excluding_first,
        },
        "delta_with_minus_without": delta,
        "ratio_with_over_without": ratio,
    }
    print(json.dumps(payload_json, indent=2))


if __name__ == "__main__":
    main()
