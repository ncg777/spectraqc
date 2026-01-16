from __future__ import annotations
import numpy as np


def _spectral_flatness(power: np.ndarray) -> float | None:
    if power.size == 0:
        return None
    power = np.asarray(power, dtype=np.float64)
    power = np.clip(power, 1e-20, None)
    geom_mean = float(np.exp(np.mean(np.log(power))))
    arith_mean = float(np.mean(power))
    if arith_mean <= 0:
        return None
    return geom_mean / arith_mean


def detect_spectral_artifacts(
    freqs: np.ndarray,
    mean_db: np.ndarray,
    *,
    expected_max_hz: float,
    config: dict | None = None
) -> dict:
    """
    Detect spectral brickwall cutoffs and mirrored spectral images (upsampling).

    Args:
        freqs: Frequency grid in Hz.
        mean_db: LTPSD mean in dB on the same grid.
        expected_max_hz: Declared maximum frequency for the profile.
        config: Optional configuration dictionary with "cutoff" and "mirror" blocks.

    Returns:
        Dictionary of measurements for downstream threshold evaluation.
    """
    cfg = config or {}
    cutoff_cfg = cfg.get("cutoff", {})
    mirror_cfg = cfg.get("mirror", {})

    freqs = np.asarray(freqs, dtype=np.float64)
    mean_db = np.asarray(mean_db, dtype=np.float64)

    min_hz = float(cutoff_cfg.get("min_hz", 4000.0))
    drop_db = float(cutoff_cfg.get("drop_db", 24.0))
    window_bins = int(cutoff_cfg.get("window_bins", 6))
    hold_bins = int(cutoff_cfg.get("hold_bins", 8))

    cutoff_freq = None
    cutoff_drop_db = None
    idx_start = int(np.searchsorted(freqs, min_hz))
    if idx_start < len(freqs) - hold_bins and window_bins > 0:
        for i in range(idx_start, len(freqs) - hold_bins):
            start = max(0, i - window_bins)
            if i <= start:
                continue
            baseline = float(np.median(mean_db[start:i]))
            if baseline - mean_db[i] < drop_db:
                continue
            segment = mean_db[i:i + hold_bins]
            if np.all(segment < baseline - drop_db):
                cutoff_freq = float(freqs[i])
                cutoff_drop_db = float(baseline - float(np.median(segment)))
                break

    mirror_similarity = None
    mirror_flatness = None
    mirror_pivot_hz = None
    mirror_band_hz = None
    min_bins = int(mirror_cfg.get("min_bins", 8))
    if cutoff_freq is not None:
        pivot = cutoff_freq
        mirror_pivot_hz = float(pivot)
        hi_mask = (freqs > pivot) & (freqs <= expected_max_hz)
        if np.count_nonzero(hi_mask) >= min_bins:
            freqs_hi = freqs[hi_mask]
            mean_hi = mean_db[hi_mask]
            freqs_mirror = 2.0 * pivot - freqs_hi
            valid = freqs_mirror >= freqs[0]
            freqs_hi = freqs_hi[valid]
            mean_hi = mean_hi[valid]
            freqs_mirror = freqs_mirror[valid]
            if freqs_hi.size >= min_bins:
                mean_lo = np.interp(freqs_mirror, freqs, mean_db)
                if float(np.std(mean_hi)) > 0 and float(np.std(mean_lo)) > 0:
                    mirror_similarity = float(np.corrcoef(mean_hi, mean_lo)[0, 1])
                mirror_band_hz = (float(freqs_hi[0]), float(freqs_hi[-1]))
                mirror_flatness = _spectral_flatness(10.0 ** (mean_hi / 10.0))

    return {
        "cutoff_freq_hz": cutoff_freq,
        "cutoff_drop_db": cutoff_drop_db,
        "expected_max_hz": float(expected_max_hz),
        "mirror_similarity": mirror_similarity,
        "mirror_flatness": mirror_flatness,
        "mirror_pivot_hz": mirror_pivot_hz,
        "mirror_band_hz": mirror_band_hz,
    }
