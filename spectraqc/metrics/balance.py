"""Stereo balance metrics."""
from __future__ import annotations

import numpy as np

from spectraqc.metrics.levels import rms_dbfs_mono
from spectraqc.metrics.loudness import integrated_lufs_mono


def _validate_multichannel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected 2D audio array for balance metrics.")
    if x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError("Expected at least two channels for balance metrics.")
    return x


def _finite_or_none(value: float) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _channel_labels(count: int) -> list[str]:
    labels = []
    for idx in range(count):
        if idx == 0:
            labels.append("L")
        elif idx == 1:
            labels.append("R")
        else:
            labels.append(f"ch{idx + 1}")
    return labels


def balance_metrics(x: np.ndarray, fs: float) -> dict:
    """Compute per-channel RMS/LUFS metrics and channel balance deltas."""
    x = _validate_multichannel(x)
    channel_count = x.shape[1]

    rms_values: list[float | None] = []
    lufs_values: list[float | None] = []
    for idx in range(channel_count):
        rms_values.append(_finite_or_none(rms_dbfs_mono(x[:, idx])))
        try:
            lufs_values.append(_finite_or_none(integrated_lufs_mono(x[:, idx], fs)))
        except Exception:
            lufs_values.append(None)

    rms_delta = None
    lufs_delta = None
    if channel_count >= 2:
        left_rms = rms_values[0]
        right_rms = rms_values[1]
        if left_rms is not None and right_rms is not None:
            rms_delta = float(left_rms - right_rms)
        left_lufs = lufs_values[0]
        right_lufs = lufs_values[1]
        if left_lufs is not None and right_lufs is not None:
            lufs_delta = float(left_lufs - right_lufs)

    labels = _channel_labels(channel_count)
    channels = [
        {
            "index": idx,
            "label": labels[idx],
            "rms_dbfs": rms_values[idx],
            "lufs_i": lufs_values[idx],
        }
        for idx in range(channel_count)
    ]

    return {
        "channels": channels,
        "rms_delta_db": rms_delta,
        "rms_delta_abs_db": None if rms_delta is None else abs(rms_delta),
        "lufs_delta_lu": lufs_delta,
        "lufs_delta_abs_lu": None if lufs_delta is None else abs(lufs_delta),
    }
