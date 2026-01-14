"""True peak measurement module."""
from __future__ import annotations
import numpy as np


def _kaiser_sinc_filter(
    oversample: int,
    *,
    num_taps: int | None = None,
    beta: float = 8.6
) -> np.ndarray:
    """Design a Kaiser-windowed sinc low-pass filter for oversampling."""
    if oversample <= 0:
        raise ValueError("oversample must be positive.")
    taps = num_taps if num_taps is not None else (64 * oversample + 1)
    if taps % 2 == 0:
        taps += 1
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    h = np.sinc(n / oversample)
    window = np.kaiser(taps, beta)
    h *= window
    h /= np.sum(h)
    h *= oversample
    return h.astype(np.float64)


def true_peak_dbtp_mono(
    x: np.ndarray,
    fs: float,
    oversample: int = 4
) -> float:
    """
    Compute true peak level in dBTP for mono audio using sinc oversampling.

    Uses Kaiser-windowed sinc reconstruction with 4x oversampling.
    """
    _ = fs
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("true_peak_dbtp_mono expects 1D mono audio.")
    if x.size == 0:
        raise ValueError("true_peak_dbtp_mono expects non-empty audio.")
    if oversample <= 0:
        raise ValueError("oversample must be positive.")

    up = int(oversample)
    y = np.zeros(x.size * up, dtype=np.float64)
    y[::up] = x
    h = _kaiser_sinc_filter(up)
    y_filt = np.convolve(y, h, mode="same")
    peak = float(np.max(np.abs(y_filt)))
    if peak <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))
