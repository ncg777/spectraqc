"""Loudness measurement module."""
from __future__ import annotations
import numpy as np


def integrated_lufs_mono(x: np.ndarray, fs: float) -> float:
    """
    Compute integrated loudness in LUFS for mono audio.
    
    Uses BS.1770 algorithm via pyloudnorm.
    
    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz
        
    Returns:
        Integrated loudness in LUFS
    """
    import pyloudnorm as pyln
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("integrated_lufs_mono expects 1D mono audio.")
    
    meter = pyln.Meter(int(fs))
    return float(meter.integrated_loudness(x))
