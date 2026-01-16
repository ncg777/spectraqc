"""Dynamic range metrics."""
from __future__ import annotations

import numpy as np

from spectraqc.metrics.loudness import short_term_lufs_series_mono


def _validate_mono(x: np.ndarray) -> np.ndarray:
    """Validate and coerce mono audio arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        raise ValueError("Expected non-empty audio array.")
    return x


def _frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive.")
    if x.size < frame_size:
        return x.reshape(1, -1)
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_size)[::hop_size]
    if frames.size == 0:
        return x.reshape(1, -1)
    return frames


def _rms_dbfs(frames: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(frames ** 2, axis=-1))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12))


def dynamic_range_percentile_dbfs_mono(
    x: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 3.0,
    hop_seconds: float = 1.0,
    low_percentile: float = 10.0,
    high_percentile: float = 95.0,
) -> float:
    """
    Compute dynamic range from percentile-based RMS distribution (dBFS).

    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz
        frame_seconds: Frame size in seconds
        hop_seconds: Hop size in seconds
        low_percentile: Low percentile for DR calculation
        high_percentile: High percentile for DR calculation

    Returns:
        Dynamic range in dB (high - low percentile)
    """
    x = _validate_mono(x)
    fs = float(fs)
    frame_size = int(round(frame_seconds * fs))
    hop_size = int(round(hop_seconds * fs))
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_seconds and hop_seconds must be > 0.")
    frames = _frame_signal(x, frame_size, hop_size)
    rms_db = _rms_dbfs(frames)
    finite = rms_db[np.isfinite(rms_db)]
    if finite.size == 0:
        return float("-inf")
    low = np.percentile(finite, low_percentile)
    high = np.percentile(finite, high_percentile)
    return float(high - low)


def dynamic_range_short_term_lufs_mono(
    x: np.ndarray,
    fs: float,
    *,
    low_percentile: float = 10.0,
    high_percentile: float = 95.0,
) -> float:
    """
    Compute dynamic range from EBU R128 short-term loudness distribution.

    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz
        low_percentile: Low percentile for DR calculation
        high_percentile: High percentile for DR calculation

    Returns:
        Dynamic range in LU (high - low percentile)
    """
    short_term = short_term_lufs_series_mono(x, fs)
    finite = short_term[np.isfinite(short_term)]
    if finite.size == 0:
        return float("-inf")
    low = np.percentile(finite, low_percentile)
    high = np.percentile(finite, high_percentile)
    return float(high - low)
