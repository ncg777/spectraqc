"""Channel consistency metrics for mono vs stereo declarations."""
from __future__ import annotations

import numpy as np

from spectraqc.metrics.correlation import correlation_summary
from spectraqc.metrics.levels import rms_dbfs_mono


def channel_consistency_metrics(
    samples: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> dict:
    """Compute correlation + mid/side RMS metrics for stereo channels."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected stereo samples with shape (n, 2).")
    if x.size == 0:
        return {
            "corr_mean": None,
            "corr_min": None,
            "corr_count": 0,
            "mid_rms_dbfs": None,
            "side_rms_dbfs": None,
            "side_mid_ratio_db": None,
        }

    corr_stats = correlation_summary(
        x,
        fs,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
    )
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    mid_rms_dbfs = rms_dbfs_mono(mid)
    side_rms_dbfs = rms_dbfs_mono(side)
    side_mid_ratio_db = None
    if np.isfinite(mid_rms_dbfs) and np.isfinite(side_rms_dbfs):
        side_mid_ratio_db = float(side_rms_dbfs - mid_rms_dbfs)
    elif np.isfinite(mid_rms_dbfs) and np.isneginf(side_rms_dbfs):
        side_mid_ratio_db = float("-inf")

    return {
        "corr_mean": corr_stats.get("mean"),
        "corr_min": corr_stats.get("min"),
        "corr_count": corr_stats.get("count"),
        "mid_rms_dbfs": mid_rms_dbfs,
        "side_rms_dbfs": side_rms_dbfs,
        "side_mid_ratio_db": side_mid_ratio_db,
    }
