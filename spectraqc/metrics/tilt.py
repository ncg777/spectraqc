from __future__ import annotations
import numpy as np


def spectral_tilt_db_per_oct(
    freqs: np.ndarray,
    mean_db: np.ndarray,
    f_low: float = 50.0,
    f_high: float = 16000.0
) -> float:
    """
    Compute spectral tilt via linear regression of dB vs log2(f).
    
    Args:
        freqs: Frequency grid in Hz
        mean_db: Mean PSD in dB
        f_low: Lower frequency bound for regression
        f_high: Upper frequency bound for regression
        
    Returns:
        Spectral tilt in dB per octave
    """
    f = np.asarray(freqs, dtype=np.float64)
    y = np.asarray(mean_db, dtype=np.float64)
    
    # Select frequency range
    m = (f >= f_low) & (f <= f_high)
    f2 = f[m]
    y2 = y[m]
    
    if f2.size < 10:
        return 0.0
    
    # Linear regression: y = slope * log2(f) + intercept
    x = np.log2(f2)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y2, rcond=None)[0]
    
    return float(slope)
