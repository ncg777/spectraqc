from __future__ import annotations
import numpy as np


def interp_to_grid(freqs_src: np.ndarray, y_src: np.ndarray, freqs_dst: np.ndarray) -> np.ndarray:
    """Interpolate y_src from freqs_src onto freqs_dst grid (clamped at edges)."""
    freqs_src = np.asarray(freqs_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    freqs_dst = np.asarray(freqs_dst, dtype=np.float64)
    if freqs_src.ndim != 1 or y_src.ndim != 1 or freqs_dst.ndim != 1:
        raise ValueError("interp_to_grid expects 1D arrays.")
    if freqs_src.size != y_src.size:
        raise ValueError("freqs_src and y_src must have same length.")
    left = float(y_src[0])
    right = float(y_src[-1])
    return np.interp(freqs_dst, freqs_src, y_src, left=left, right=right).astype(np.float64)


def interp_var_ratio(freqs_src: np.ndarray, var_src: np.ndarray, freqs_dst: np.ndarray) -> np.ndarray:
    """Interpolate variance data onto destination grid, ensuring non-negative."""
    v = interp_to_grid(freqs_src, var_src, freqs_dst)
    return np.maximum(v, 0.0)
