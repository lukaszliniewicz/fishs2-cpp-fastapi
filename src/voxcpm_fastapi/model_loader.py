from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .settings import settings

if TYPE_CHECKING:
    from voxcpm import VoxCPM

logger = logging.getLogger(__name__)

try:
    from voxcpm import VoxCPM

    HAS_VOXCPM = True
except Exception:
    VoxCPM = None  # type: ignore[assignment]
    HAS_VOXCPM = False


def is_voxcpm2_model(model_id: str) -> bool:
    return "voxcpm2" in model_id.strip().lower()


def _resolve_device() -> str:
    desired = settings.device.strip().lower()
    if desired and desired != "auto":
        return desired

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Auto-detected CUDA device for VoxCPM")
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            logger.info("Auto-detected MPS device for VoxCPM")
            return "mps"
    except Exception:
        logger.debug("Torch unavailable during device auto-detection", exc_info=True)

    return "cpu"


class VoxCPMWrapper:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = _resolve_device()
        self._model: VoxCPM | None = None
        self._loaded = False

    @property
    def model(self) -> VoxCPM:
        if self._model is None:
            raise RuntimeError("VoxCPM model is not loaded")
        return self._model

    @property
    def sample_rate(self) -> int:
        tts_model = getattr(self.model, "tts_model", None)
        value = getattr(tts_model, "sample_rate", None)
        if isinstance(value, int) and value > 0:
            return value
        return 48000

    def load(self) -> None:
        if self._loaded and self._model is not None:
            return

        if not HAS_VOXCPM or VoxCPM is None:
            raise ImportError("voxcpm is not installed. Install it with `pip install voxcpm==2.0.3`.")

        logger.info(
            "Loading VoxCPM model '%s' (device=%s optimize=%s denoiser=%s)",
            self.model_id,
            self.device,
            settings.optimize,
            settings.load_denoiser,
        )
        self._model = VoxCPM.from_pretrained(
            hf_model_id=self.model_id,
            load_denoiser=settings.load_denoiser,
            optimize=settings.optimize,
            device=self.device,
        )
        self._loaded = True

    def generate(self, **kwargs) -> np.ndarray:
        self.load()
        result = self.model.generate(**kwargs)
        return np.asarray(result, dtype=np.float32)
