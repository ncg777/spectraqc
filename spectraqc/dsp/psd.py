from __future__ import annotations
import numpy as np
from spectraqc.dsp.windowing import hann, window_power_norm


def welch_psd_db(
    x: np.ndarray,
    fs: float,
    nfft: int = 4096,
    hop: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Welch PSD estimate in dB.
    
    Uses Hann window, one-sided spectrum, power/Hz normalization.
    
    Args:
        x: Mono input signal (1D array)
        fs: Sample rate in Hz
        nfft: FFT size
        hop: Hop size between frames
        
    Returns:
        Tuple of (freqs, mean_db, var_db2):
        - freqs: Frequency bins in Hz
        - mean_db: Mean PSD in dB across frames
        - var_db2: Variance of PSD in dB² across frames
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("welch_psd_db expects mono 1D signal.")
    if nfft <= 0 or hop <= 0:
        raise ValueError("nfft and hop must be positive.")
    if len(x) < nfft:
        raise ValueError("Signal too short for given nfft.")

    w = hann(nfft).astype(np.float64)
    U = window_power_norm(w)
    eps = 1e-20

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    n_bins = freqs.size

    # One-sided correction: double all bins except DC and Nyquist
    has_nyquist = (nfft % 2 == 0)
    one_sided_gain = np.ones(n_bins, dtype=np.float64)
    if has_nyquist:
        if n_bins > 2:
            one_sided_gain[1:-1] = 2.0
    else:
        if n_bins > 1:
            one_sided_gain[1:] = 2.0

    frames_db = []
    denom = (fs * nfft * U) + eps

    for start in range(0, len(x) - nfft + 1, hop):
        seg = x[start:start + nfft] * w
        X = np.fft.rfft(seg, n=nfft)
        Pxx = (np.abs(X) ** 2) / denom
        Pxx *= one_sided_gain
        Pxx = np.maximum(Pxx, eps)
        frames_db.append(10.0 * np.log10(Pxx))

    if not frames_db:
        raise ValueError("No frames were produced (check hop/nfft).")

    D = np.stack(frames_db, axis=0)
    mean_db = np.mean(D, axis=0)
    var_db2 = np.var(D, axis=0, ddof=0)
    
    return freqs.astype(np.float64), mean_db.astype(np.float64), var_db2.astype(np.float64)
