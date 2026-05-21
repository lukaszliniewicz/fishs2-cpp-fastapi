# FishS2 CPP FastAPI

OpenAI-compatible TTS wrapper for **Fish Audio S2 Pro GGUF**, built for Pandrator-style workflows.

This server keeps the same endpoint shape used by our XTTS-style wrappers while running inference through `s2.dll` from FishS2Sharp runtime bundles (which wrap the local `s2.cpp` runtime).

## Upstream Sources

- Fish Audio open-source model/project: `https://github.com/fishaudio/fish-speech`
- Fish Audio docs: `https://docs.fish.audio`
- `s2.cpp` local C++ runtime: `https://github.com/rodrigomatta/s2.cpp`
- FishS2Sharp C# wrapper/runtime bundles: `https://github.com/subspecs/FishS2Sharp`
- S2 Pro GGUF model variants used by this wrapper: `https://huggingface.co/rodrigomt/s2-pro-gguf`

## Quick Start

```bash
# Windows
run.bat

# Linux / macOS
bash run.sh
```

The launcher starts the API at:

- `http://0.0.0.0:8020`

## Runtime Requirements

- CUDA-capable NVIDIA GPU (server is CUDA-first)
- Internet access on first run (for runtime/model downloads)

On startup, `run.py` now auto-downloads and keeps everything local:

- FishS2Sharp runtime bundle to `runtime/fishs2sharp/`
- S2 GGUF model to `models/s2-pro-q8_0.gguf` by default (use `--model-q4` for a smaller `q4_k_m` download)
- tokenizer to `models/tokenizer.json`

Useful bootstrap flags:

- `--skip-downloads` (offline mode, use existing local files)
- `--force-downloads` (refresh local artifacts)
- `--model-q4` (explicit shortcut for `q4_k_m` model download)
- `--model-quant <quant>` (choose quant: `f16`, `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`, `q3_k`, `q2_k`)
- `--n-gpu-layers` (set explicit transformer layer offload count; `-1` keeps runtime default)

You can override artifact paths and sources with env vars:

- `FISHS2_RUNTIME_DIR`
- `FISHS2_S2_DLL_PATH`
- `FISHS2_MODEL_PATH`
- `FISHS2_TOKENIZER_PATH`
- `FISHS2_RUNTIME_ZIP_URL`
- `FISHS2_RUNTIME_ZIP_SHA256`
- `FISHS2_MODEL_URL`
- `FISHS2_TOKENIZER_URL`
- `FISHS2_MODEL_SHA256`
- `FISHS2_TOKENIZER_SHA256`
- `FISHS2_HF_REPO_ID` (default: `rodrigomt/s2-pro-gguf`)
- `FISHS2_MODEL_QUANT` (default: `q8_0`)
- `FISHS2_SKIP_DOWNLOADS=true`
- `FISHS2_FORCE_DOWNLOADS=true`
- `FISHS2_N_GPU_LAYERS` (default: `-1`)

## Required Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `GET /v1/audio/voices`

Compatibility aliases and fallbacks:

- `GET /v1/voices` (alias)
- `POST /v1/audio/voices` (voice upload)
- `POST /v1/files` (legacy upload fallback)
- `GET /v1/files` (legacy voice discovery fallback)
- `DELETE /v1/voices/{voice_id}` (optional cleanup)

## Model Policy

`GET /v1/models` returns Fish S2 entries based on settings:

- `FISHS2_DEFAULT_MODEL` (default: `fishaudio/s2-pro`)
- `FISHS2_MODEL_ALIASES` (default: `fishs2,fish-s2,s2-pro`)

All listed aliases map to one backend runtime instance.

## Speech API Notes

`POST /v1/audio/speech` accepts OpenAI-style fields:

- `model`
- `input`
- `voice`
- `response_format`
- `speed` (must be `1.0`; FishS2 backend does not support speed control)
- `instructions`

And wrapper extension fields:

- `fishs2` object (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `min_tokens_before_end`, `n_threads`, `verbose`)
- `reference_audio` / aliases (`prompt_audio`, `ref_audio`) for Fish-style reference path
- `reference_text` / alias (`ref_text`) for Fish-style transcript
- `speaker_wav` (explicit local reference audio path list)
- `prompt_text` (reference transcript)
- `control` (prepended as `(control)...`)

For Fish/S2, what matters is whether a reference audio is provided, and if so, a matching transcript is required.

### Output Format

V1 supports **`wav` only**.

## Practical API Examples

Assume server is running at `http://127.0.0.1:8020`.

### 1) Health and model discovery

```bash
curl http://127.0.0.1:8020/health
curl http://127.0.0.1:8020/v1/models
```

### 2) Basic generation (default/random voice)

```bash
curl -X POST http://127.0.0.1:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishs2",
    "input": "Hello from Fish S2 CPP FastAPI.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output outputs/basic.wav
```

### 3) Upload a voice profile (stored under `voices/<voice_id>/`)

```bash
curl -X POST http://127.0.0.1:8020/v1/audio/voices \
  -F "files=@voices/sample_male/sample_male_new.wav" \
  -F "voice_id=sample_male" \
  -F "prompt_text=This is a sample transcript for the reference clip."
```

List stored voices:

```bash
curl http://127.0.0.1:8020/v1/audio/voices
```

### 4) Generate speech using a stored voice profile

```bash
curl -X POST http://127.0.0.1:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishs2",
    "input": "This line uses the uploaded sample_male profile.",
    "voice": "sample_male",
    "response_format": "wav"
  }' \
  --output outputs/from_profile.wav
```

### 5) Direct one-off cloning with explicit reference audio + transcript

`reference_audio` is a local file path on the server machine.

```bash
curl -X POST http://127.0.0.1:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishs2",
    "input": "This line clones directly from a reference file.",
    "voice": "default",
    "reference_audio": "voices/sample_male/sample_male_new.wav",
    "reference_text": "This is a sample transcript for the reference clip.",
    "response_format": "wav"
  }' \
  --output outputs/direct_clone.wav
```

### 6) Advanced generation controls (`fishs2` overrides)

```bash
curl -X POST http://127.0.0.1:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishs2",
    "input": "Custom sampling settings example.",
    "voice": "sample_male",
    "response_format": "wav",
    "fishs2": {
      "max_new_tokens": 900,
      "temperature": 0.7,
      "top_p": 0.85,
      "top_k": 40,
      "min_tokens_before_end": 0,
      "n_threads": 0,
      "verbose": true
    }
  }' \
  --output outputs/custom_params.wav
```

### 7) Delete an uploaded voice profile

```bash
curl -X DELETE http://127.0.0.1:8020/v1/voices/sample_male
```

## Voice Upload Notes

Upload fields accepted for compatibility:

- `files` (multi-part list)
- `audio_sample` (single file)
- `file` (legacy single file)

Metadata fields:

- `voice_id`
- `name`
- `purpose`
- `prompt_text`

Voice data is stored under `voices/<voice_id>/` with a `meta.json` file.

When using a stored voice for synthesis, the first uploaded audio sample is used as reference audio.

## NVIDIA / CUDA Notes

- `run.py` blocks startup when `--backend cuda` is selected and no NVIDIA GPU is detected.
- To bypass this check intentionally, use `--skip-gpu-check` (or set `FISHS2_SKIP_GPU_CHECK=true`).
- On Windows, the runtime must be able to load CUDA dependencies (`nvcuda.dll`, `cudart64_12.dll`, `cublas64_12.dll`).
- `FISHS2_N_GPU_LAYERS` controls transformer GPU offload (`-1` keeps runtime default behavior, typically full offload on GPU backends).

## Local Cache Policy

The bootstrapper forces local, portable cache locations inside this repo:

- `.hf/` for Hugging Face caches (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, etc.)
- `.pip-cache/` for pip
- `.pixi-cache/` for pixi
- `.tmp/` for temporary downloads

This prevents fallback to user-global cache folders on the host machine.

Optional DLL search path overrides:

- `FISHS2_RUNTIME_EXTRA_DLL_DIRS` (comma or semicolon separated)

## Dev

```bash
bin\pixi run pytest
```
