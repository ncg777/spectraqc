"""Clipping detection metrics."""
from __future__ import annotations

import numpy as np


def _validate_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        raise ValueError("Expected non-empty audio array.")
    return x


def _run_lengths(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=np.int64)
    gaps = np.flatnonzero(np.diff(indices) > 1)
    run_starts = np.concatenate(([0], gaps + 1))
    run_ends = np.concatenate((gaps, [indices.size - 1]))
    return indices[run_ends] - indices[run_starts] + 1


def _clipping_stats_mono(x: np.ndarray, *, atol: float) -> dict:
    x = _validate_mono(x)
    abs_x = np.abs(x)
    max_amp = float(np.max(abs_x))
    if max_amp <= 0:
        return {
            "max_amplitude": max_amp,
            "total_samples": int(x.size),
            "clipped_samples": 0,
            "clipped_ratio": 0.0,
            "run_count": 0,
            "max_run_length": 0,
        }
    mask = np.isclose(abs_x, max_amp, atol=float(atol), rtol=0.0)
    clipped_samples = int(np.sum(mask))
    run_lengths = _run_lengths(np.flatnonzero(mask))
    run_count = int(run_lengths.size)
    max_run_length = int(np.max(run_lengths)) if run_count else 0
    clipped_ratio = float(clipped_samples / x.size)
    return {
        "max_amplitude": max_amp,
        "total_samples": int(x.size),
        "clipped_samples": clipped_samples,
        "clipped_ratio": clipped_ratio,
        "run_count": run_count,
        "max_run_length": max_run_length,
    }


def detect_clipping_runs(
    samples: np.ndarray,
    *,
    atol: float = 1e-6,
) -> dict:
    """Detect sample-level clipping runs at max amplitude."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        return {"policy": "mono", **_clipping_stats_mono(samples, atol=atol)}

    if samples.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")

    channel_reports = []
    total_samples = int(samples.shape[0] * samples.shape[1])
    clipped_samples = 0
    run_count = 0
    max_run_length = 0
    max_amplitude = 0.0
    for idx in range(samples.shape[1]):
        stats = _clipping_stats_mono(samples[:, idx], atol=atol)
        channel_reports.append({"channel_index": int(idx), **stats})
        clipped_samples += int(stats["clipped_samples"])
        run_count += int(stats["run_count"])
        max_run_length = max(max_run_length, int(stats["max_run_length"]))
        max_amplitude = max(max_amplitude, float(stats["max_amplitude"]))

    clipped_ratio = float(clipped_samples / total_samples) if total_samples else 0.0
    return {
        "policy": "per_channel",
        "max_amplitude": max_amplitude,
        "total_samples": total_samples,
        "clipped_samples": clipped_samples,
        "clipped_ratio": clipped_ratio,
        "run_count": run_count,
        "max_run_length": max_run_length,
        "channels": channel_reports,
    }
