"""Transient spike detection using high-pass filtering and derivative thresholds."""
from __future__ import annotations

import numpy as np


def _highpass_filter(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return np.asarray(x, dtype=np.float64)
    if fs <= 0:
        raise ValueError("Sample rate must be positive.")
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def _pick_spike_indices(indices: np.ndarray, values: np.ndarray, min_sep: int) -> list[int]:
    if indices.size == 0:
        return []
    if min_sep <= 1:
        return indices.astype(int).tolist()
    keep: list[int] = []
    current_idx = int(indices[0])
    current_val = float(values[current_idx])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - current_idx <= min_sep:
            val = float(values[idx])
            if val > current_val:
                current_idx = idx
                current_val = val
        else:
            keep.append(current_idx)
            current_idx = idx
            current_val = float(values[idx])
    keep.append(current_idx)
    return keep


def _detect_spikes_mono(
    x: np.ndarray,
    fs: float,
    *,
    highpass_hz: float,
    derivative_threshold: float,
    min_separation_seconds: float,
) -> list[int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        return []
    if derivative_threshold <= 0:
        raise ValueError("derivative_threshold must be positive.")
    if min_separation_seconds < 0:
        raise ValueError("min_separation_seconds must be non-negative.")

    hp = _highpass_filter(x, fs, float(highpass_hz))
    deriv = np.diff(hp, prepend=hp[0])
    magnitude = np.abs(deriv)
    indices = np.flatnonzero(magnitude >= float(derivative_threshold))
    min_sep = int(round(min_separation_seconds * fs))
    return _pick_spike_indices(indices, magnitude, min_sep)


def _summarize_spikes(indices: list[int], fs: float) -> dict:
    times = [float(idx / fs) for idx in indices]
    return {
        "count": int(len(indices)),
        "sample_indices": [int(i) for i in indices],
        "time_indices_s": times,
    }


def detect_transient_spikes(
    samples: np.ndarray,
    fs: float,
    *,
    config: dict | None = None,
) -> dict:
    """
    Detect transient spikes using a high-pass filter plus derivative thresholding.

    Args:
        samples: Mono or stereo audio samples.
        fs: Sample rate.
        config: Configuration dict with thresholds and channel policy.

    Returns:
        Dictionary with policy, counts, and time indices.
    """
    cfg = config or {}
    policy = str(cfg.get("channel_policy", "per_channel")).strip().lower()
    highpass_hz = float(cfg.get("highpass_hz", 2000.0))
    derivative_threshold = float(cfg.get("derivative_threshold", 0.2))
    min_sep = float(cfg.get("min_separation_seconds", 0.005))

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        indices = _detect_spikes_mono(
            samples,
            fs,
            highpass_hz=highpass_hz,
            derivative_threshold=derivative_threshold,
            min_separation_seconds=min_sep,
        )
        return {"policy": "mono", **_summarize_spikes(indices, fs)}

    if samples.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")

    if policy == "average":
        mono = np.mean(samples, axis=1)
        indices = _detect_spikes_mono(
            mono,
            fs,
            highpass_hz=highpass_hz,
            derivative_threshold=derivative_threshold,
            min_separation_seconds=min_sep,
        )
        return {"policy": "average", **_summarize_spikes(indices, fs)}

    if policy != "per_channel":
        raise ValueError("channel_policy must be 'per_channel' or 'average'.")

    channel_reports = []
    merged_indices: list[int] = []
    for idx in range(samples.shape[1]):
        channel = samples[:, idx]
        indices = _detect_spikes_mono(
            channel,
            fs,
            highpass_hz=highpass_hz,
            derivative_threshold=derivative_threshold,
            min_separation_seconds=min_sep,
        )
        merged_indices.extend(indices)
        channel_reports.append(
            {"channel_index": int(idx), **_summarize_spikes(indices, fs)}
        )
    merged_indices = sorted(set(merged_indices))
    return {
        "policy": "per_channel",
        **_summarize_spikes(merged_indices, fs),
        "channels": channel_reports,
    }
