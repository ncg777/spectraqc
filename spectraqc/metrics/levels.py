"""Peak, RMS, and crest factor metrics."""
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


def peak_dbfs_mono(x: np.ndarray) -> float:
    """Compute sample peak in dBFS for mono audio."""
    x = _validate_mono(x)
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def rms_dbfs_mono(x: np.ndarray) -> float:
    """Compute RMS level in dBFS for mono audio."""
    x = _validate_mono(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= 0:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def crest_factor_db_mono(x: np.ndarray) -> float:
    """Compute crest factor in dB (peak-to-RMS ratio) for mono audio."""
    x = _validate_mono(x)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    if peak <= 0 or rms <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak / rms))
