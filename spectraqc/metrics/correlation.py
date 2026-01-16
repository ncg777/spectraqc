"""Stereo correlation metrics."""
from __future__ import annotations

import numpy as np


def windowed_correlation_coefficients(
    samples: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> np.ndarray:
    """
    Compute windowed Pearson correlation coefficients for stereo audio.

    Args:
        samples: Stereo audio samples shaped (n, 2).
        fs: Sample rate in Hz.
        frame_seconds: Window length in seconds.
        hop_seconds: Hop length in seconds.

    Returns:
        Array of correlation coefficients per window.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected stereo samples with shape (n, 2).")
    if x.size == 0:
        return np.array([], dtype=np.float64)
    frame_size = int(round(float(frame_seconds) * float(fs)))
    hop_size = int(round(float(hop_seconds) * float(fs)))
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_seconds and hop_seconds must be > 0.")

    if x.shape[0] < frame_size:
        frames = x.reshape(1, -1, 2)
    else:
        frames = np.lib.stride_tricks.sliding_window_view(
            x, frame_size, axis=0
        )[::hop_size]
        if frames.size == 0:
            frames = x.reshape(1, -1, 2)

    left = frames[..., 0]
    right = frames[..., 1]
    left_centered = left - np.mean(left, axis=-1, keepdims=True)
    right_centered = right - np.mean(right, axis=-1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=-1)
    denom = np.sqrt(
        np.sum(left_centered ** 2, axis=-1) * np.sum(right_centered ** 2, axis=-1)
    )
    corr = np.divide(
        numerator,
        denom,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denom > 0,
    )
    return corr.astype(np.float64)


def correlation_summary(
    samples: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> dict:
    """Return summary stats for windowed stereo correlation."""
    corr = windowed_correlation_coefficients(
        samples,
        fs,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
    )
    if corr.size == 0:
        return {"mean": None, "min": None, "count": 0}
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        return {"mean": None, "min": None, "count": 0}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "count": int(finite.size),
    }


def estimate_interchannel_delay(
    samples: np.ndarray,
    fs: float,
    *,
    max_delay_seconds: float = 0.01,
) -> dict:
    """
    Estimate inter-channel delay using cross-correlation for stereo audio.

    Args:
        samples: Stereo audio samples shaped (n, 2).
        fs: Sample rate in Hz.
        max_delay_seconds: Maximum absolute delay to search in seconds.

    Returns:
        Dictionary with delay in seconds/samples and correlation value.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected stereo samples with shape (n, 2).")
    if x.size == 0:
        return {"delay_seconds": None, "delay_samples": None, "correlation": None}
    max_delay_seconds = float(max_delay_seconds)
    if max_delay_seconds <= 0:
        raise ValueError("max_delay_seconds must be > 0.")

    left = x[:, 0] - np.mean(x[:, 0])
    right = x[:, 1] - np.mean(x[:, 1])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    corr = np.correlate(left, right, mode="full")
    if denom > 0:
        corr = corr / denom
    else:
        corr = np.zeros_like(corr, dtype=np.float64)

    n = x.shape[0]
    lags = np.arange(-n + 1, n, dtype=np.int64)
    max_delay_samples = int(round(max_delay_seconds * float(fs)))
    if max_delay_samples < 1:
        max_delay_samples = 1
    window = np.abs(lags) <= max_delay_samples
    if not np.any(window):
        return {"delay_seconds": None, "delay_samples": None, "correlation": None}
    corr_window = corr[window]
    lags_window = lags[window]
    idx = int(np.argmax(np.abs(corr_window)))
    lag = int(lags_window[idx])
    delay_seconds = float(lag) / float(fs)
    return {
        "delay_seconds": delay_seconds,
        "delay_samples": int(lag),
        "correlation": float(corr_window[idx]),
    }
