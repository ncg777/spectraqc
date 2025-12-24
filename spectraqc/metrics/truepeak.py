"""True peak measurement module."""
from __future__ import annotations
import numpy as np


def _kaiser_beta(att_db: float) -> float:
    """Compute Kaiser window beta parameter for given stopband attenuation."""
    if att_db > 50:
        return 0.1102 * (att_db - 8.7)
    if att_db >= 21:
        return 0.5842 * (att_db - 21) ** 0.4 + 0.07886 * (att_db - 21)
    return 0.0


def _sinc_lowpass_fir(num_taps: int, cutoff: float, att_db: float = 80.0) -> np.ndarray:
    """Design a windowed-sinc lowpass FIR filter."""
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for symmetric FIR.")
    M = num_taps - 1
    n = np.arange(num_taps, dtype=np.float64)
    fc = float(cutoff)
    m = n - M / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * m)
    beta = _kaiser_beta(att_db)
    w = np.kaiser(num_taps, beta).astype(np.float64)
    h *= w
    h /= np.sum(h)
    return h.astype(np.float64)


def _upsample_zeros(x: np.ndarray, factor: int) -> np.ndarray:
    """Upsample by inserting zeros."""
    y = np.zeros(x.size * factor, dtype=np.float64)
    y[::factor] = x.astype(np.float64)
    return y


def true_peak_dbtp_mono(
    x: np.ndarray,
    fs: float,
    oversample: int = 4,
    *,
    fir_taps: int = 63,
    cutoff_rel_nyquist: float | None = None
) -> float:
    """
    Compute true peak level in dBTP for mono audio.
    
    Uses oversampled windowed-sinc FIR reconstruction per BS.1770.
    
    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz (for reference, not used in computation)
        oversample: Oversampling factor (default 4x per spec)
        fir_taps: Number of FIR filter taps (must be odd)
        cutoff_rel_nyquist: Lowpass cutoff relative to oversampled Nyquist
        
    Returns:
        True peak level in dBTP
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("true_peak_dbtp_mono expects 1D mono audio.")
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    
    if oversample == 1:
        peak = float(np.max(np.abs(x))) + 1e-30
        return 20.0 * np.log10(peak)
    
    if fir_taps % 2 == 0:
        fir_taps += 1
    
    if cutoff_rel_nyquist is None:
        cutoff_rel_nyquist = (0.5 / oversample) * 0.9
    
    h = _sinc_lowpass_fir(fir_taps, float(cutoff_rel_nyquist), att_db=80.0)
    y = _upsample_zeros(x, oversample)
    z = np.convolve(y, h, mode="full")
    
    # Compensate for group delay
    gd = (h.size - 1) // 2
    z = z[gd:gd + y.size]
    
    peak = float(np.max(np.abs(z))) + 1e-30
    return 20.0 * np.log10(peak)
