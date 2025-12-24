from __future__ import annotations
import numpy as np
from spectraqc.types import FrequencyBand, BandMetrics
from spectraqc.analysis.bands import band_mask


def _df_weights(freqs: np.ndarray) -> np.ndarray:
    """Compute Δf weights for frequency integration."""
    f = np.asarray(freqs, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("freqs must be 1D with at least 2 points.")
    df = np.empty_like(f)
    df[0] = f[1] - f[0]
    df[-1] = f[-1] - f[-2]
    df[1:-1] = 0.5 * (f[2:] - f[:-2])
    return np.maximum(df, 0.0)


def band_metrics(
    freqs: np.ndarray,
    delta_db: np.ndarray,
    var_db2: np.ndarray,
    ref_var_db2: np.ndarray,
    bands: list[FrequencyBand]
) -> list[BandMetrics]:
    """
    Compute band-level metrics from deviation and variance data.
    
    Args:
        freqs: Frequency grid in Hz
        delta_db: Deviation curve (input - reference) in dB
        var_db2: Input variance in dB² per bin
        ref_var_db2: Reference variance in dB² per bin
        bands: List of frequency bands to analyze
        
    Returns:
        List of BandMetrics for each band
    """
    out: list[BandMetrics] = []
    f = np.asarray(freqs, dtype=np.float64)
    d = np.asarray(delta_db, dtype=np.float64)
    v = np.asarray(var_db2, dtype=np.float64)
    rv = np.asarray(ref_var_db2, dtype=np.float64)
    
    if not (f.shape == d.shape == v.shape == rv.shape):
        raise ValueError("freqs, delta_db, var_db2, ref_var_db2 must have identical shapes.")
    
    w = _df_weights(f)
    
    for b in bands:
        m = band_mask(f, b)
        if not np.any(m):
            continue
        wm = w[m]
        wsum = float(np.sum(wm))
        if wsum <= 0:
            continue
        
        # Weighted mean deviation
        mean_dev = float(np.sum(d[m] * wm) / wsum)
        # Max absolute deviation
        max_dev = float(np.max(np.abs(d[m])))
        # Variance ratio
        v_mean = float(np.sum(v[m] * wm) / wsum)
        rv_mean = float(np.sum(rv[m] * wm) / wsum)
        vr = float(v_mean / (rv_mean + 1e-12))
        
        out.append(BandMetrics(
            band=b,
            mean_deviation_db=mean_dev,
            max_deviation_db=max_dev,
            variance_ratio=vr
        ))
    
    return out
