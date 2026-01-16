"""Noise floor estimation metrics."""
from __future__ import annotations

import numpy as np


def _validate_mono(x: np.ndarray) -> np.ndarray:
    """Validate and coerce mono audio arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        raise ValueError("Expected non-empty audio array.")
    return x


def noise_floor_dbfs_mono(
    x: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.1,
    percentile: float = 10.0,
) -> float:
    """Estimate noise floor via percentile of short-term RMS."""
    x = _validate_mono(x)
    if fs <= 0:
        raise ValueError("Sample rate must be positive.")
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive.")
    frame_len = max(1, int(round(frame_seconds * fs)))
    rms_vals = []
    for start in range(0, x.size, frame_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_vals.append(rms)
    if not rms_vals:
        return float("-inf")
    noise_rms = float(np.percentile(rms_vals, percentile))
    if noise_rms <= 0:
        return float("-inf")
    return float(20.0 * np.log10(noise_rms))
