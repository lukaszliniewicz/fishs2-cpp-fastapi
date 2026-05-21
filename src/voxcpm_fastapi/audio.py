from __future__ import annotations

import io

import numpy as np

SAMPLE_RATE = 48000
SUPPORTED_FORMATS = {"wav"}


def numpy_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    normalized = np.asarray(audio, dtype=np.float32).reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, normalized, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
