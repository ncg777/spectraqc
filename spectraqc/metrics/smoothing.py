from __future__ import annotations
import numpy as np


def smooth_octave_fraction(
    freqs_hz: np.ndarray,
    y: np.ndarray,
    octave_fraction: float = 1 / 6,
    *,
    min_hz: float = 20.0
) -> np.ndarray:
    """
    Apply log-frequency smoothing using octave-fraction boxcar average.
    
    Args:
        freqs_hz: Frequency grid in Hz (strictly increasing)
        y: Values to smooth (same length as freqs_hz)
        octave_fraction: Width of smoothing window in octaves
        min_hz: Minimum frequency to apply smoothing
        
    Returns:
        Smoothed values array
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if f.ndim != 1 or y.ndim != 1 or f.size != y.size:
        raise ValueError("smooth_octave_fraction expects 1D freqs and 1D y of same length.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("freqs_hz must be strictly increasing.")
    
    out = y.copy()
    w = float(octave_fraction)
    if w <= 0:
        return out
    
    half = 0.5 * w
    for i in range(f.size):
        fi = f[i]
        if fi <= 0 or fi < min_hz:
            continue
        lo = fi * (2.0 ** (-half))
        hi = fi * (2.0 ** (+half))
        j0 = int(np.searchsorted(f, lo, side="left"))
        j1 = int(np.searchsorted(f, hi, side="right"))
        if j1 <= j0:
            continue
        out[i] = float(np.mean(y[j0:j1]))
    return out


def smooth_log_hz(
    freqs_hz: np.ndarray,
    y: np.ndarray,
    *,
    bins_per_octave: int = 12,
    min_hz: float = 20.0
) -> np.ndarray:
    """
    Apply log-frequency smoothing using fixed log-spaced bins.

    Args:
        freqs_hz: Frequency grid in Hz (strictly increasing)
        y: Values to smooth (same length as freqs_hz)
        bins_per_octave: Number of bins per octave
        min_hz: Minimum frequency to apply smoothing

    Returns:
        Smoothed values array
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if f.ndim != 1 or y.ndim != 1 or f.size != y.size:
        raise ValueError("smooth_log_hz expects 1D freqs and 1D y of same length.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("freqs_hz must be strictly increasing.")
    if bins_per_octave <= 0:
        raise ValueError("bins_per_octave must be positive.")

    out = y.copy()
    valid = f >= max(min_hz, np.finfo(np.float64).tiny)
    if not np.any(valid):
        return out

    f_valid = f[valid]
    y_valid = y[valid]
    log_bins = np.floor(np.log2(f_valid / min_hz) * bins_per_octave).astype(int)
    unique_bins = np.unique(log_bins)
    bin_means = {}
    for b in unique_bins:
        idx = log_bins == b
        bin_means[b] = float(np.mean(y_valid[idx]))
    out[valid] = np.array([bin_means[b] for b in log_bins], dtype=np.float64)
    return out
