"""Broadband transient detection using spectral flux or RMS spikes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _FrameFeature:
    start: int
    value: float


def _frame_audio(x: np.ndarray, frame_len: int, hop_len: int) -> list[_FrameFeature]:
    frames: list[_FrameFeature] = []
    for start in range(0, x.size, hop_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        frames.append(_FrameFeature(start=start, value=float(np.sqrt(np.mean(frame ** 2)))))
    return frames


def _compute_spectral_flux(
    x: np.ndarray,
    frame_len: int,
    hop_len: int,
) -> list[_FrameFeature]:
    window = np.hanning(frame_len).astype(np.float64)
    prev_mag = None
    features: list[_FrameFeature] = []
    for start in range(0, x.size, hop_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        if frame.size < frame_len:
            padded = np.zeros(frame_len, dtype=np.float64)
            padded[:frame.size] = frame
            frame = padded
        frame = frame * window
        mag = np.abs(np.fft.rfft(frame))
        mag_sum = float(np.sum(mag))
        if mag_sum > 0:
            mag = mag / mag_sum
        if prev_mag is None:
            flux = 0.0
        else:
            diff = mag - prev_mag
            flux = float(np.sum(diff[diff > 0.0]))
        prev_mag = mag
        features.append(_FrameFeature(start=start, value=flux))
    return features


def _build_segments(
    features: list[_FrameFeature],
    *,
    fs: float,
    frame_len: int,
    total_samples: int,
    threshold: float,
    min_duration_seconds: float,
    merge_gap_seconds: float,
) -> dict:
    segments: list[dict] = []
    if not features:
        return {
            "segments": [],
            "total_duration_s": 0.0,
            "longest_duration_s": 0.0,
            "count": 0,
        }
    min_duration_seconds = max(0.0, float(min_duration_seconds))
    merge_gap_seconds = max(0.0, float(merge_gap_seconds))
    merge_gap_samples = int(round(merge_gap_seconds * fs))

    in_run = False
    run_start = 0
    run_end = 0
    peak_value = 0.0
    peak_start = 0

    for feature in features:
        start = feature.start
        end = min(start + frame_len, total_samples)
        active = feature.value >= threshold
        if active and not in_run:
            in_run = True
            run_start = start
            run_end = end
            peak_value = feature.value
            peak_start = start
        elif active:
            if start - run_end <= merge_gap_samples:
                run_end = end
            else:
                duration_s = (run_end - run_start) / fs
                if duration_s >= min_duration_seconds:
                    segments.append(
                        {
                            "start_s": float(run_start / fs),
                            "end_s": float(run_end / fs),
                            "duration_s": float(duration_s),
                            "peak_value": float(peak_value),
                            "peak_time_s": float(peak_start / fs),
                        }
                    )
                run_start = start
                run_end = end
                peak_value = feature.value
                peak_start = start
            if feature.value > peak_value:
                peak_value = feature.value
                peak_start = start
        elif in_run:
            duration_s = (run_end - run_start) / fs
            if duration_s >= min_duration_seconds:
                segments.append(
                    {
                        "start_s": float(run_start / fs),
                        "end_s": float(run_end / fs),
                        "duration_s": float(duration_s),
                        "peak_value": float(peak_value),
                        "peak_time_s": float(peak_start / fs),
                    }
                )
            in_run = False

    if in_run:
        duration_s = (run_end - run_start) / fs
        if duration_s >= min_duration_seconds:
            segments.append(
                {
                    "start_s": float(run_start / fs),
                    "end_s": float(run_end / fs),
                    "duration_s": float(duration_s),
                    "peak_value": float(peak_value),
                    "peak_time_s": float(peak_start / fs),
                }
            )

    total_duration = float(sum(seg["duration_s"] for seg in segments))
    longest = float(max((seg["duration_s"] for seg in segments), default=0.0))
    return {
        "segments": segments,
        "total_duration_s": total_duration,
        "longest_duration_s": longest,
        "count": int(len(segments)),
    }


def _select_channel(samples: np.ndarray, policy: str) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")
    if policy == "average":
        return np.mean(samples, axis=1)
    raise ValueError("channel_policy must be 'average'.")


def detect_broadband_transients(
    samples: np.ndarray,
    fs: float,
    *,
    config: dict | None = None,
) -> dict:
    """
    Detect broadband transients using spectral flux or wideband RMS spikes.

    Returns a summary with segment time spans and aggregate counts.
    """
    cfg = config or {}
    method = str(cfg.get("method", "spectral_flux")).strip().lower()
    channel_policy = str(cfg.get("channel_policy", "average")).strip().lower()
    frame_seconds = float(cfg.get("frame_seconds", 0.05))
    hop_seconds = float(cfg.get("hop_seconds", 0.02))
    min_duration_seconds = float(cfg.get("min_duration_seconds", 0.02))
    merge_gap_seconds = float(cfg.get("merge_gap_seconds", 0.02))

    x = _select_channel(np.asarray(samples, dtype=np.float64), channel_policy)
    if x.size == 0 or fs <= 0:
        return {
            "method": method,
            "channel_policy": channel_policy,
            "segments": [],
            "total_duration_s": 0.0,
            "longest_duration_s": 0.0,
            "count": 0,
        }

    frame_len = max(1, int(round(frame_seconds * fs)))
    hop_len = max(1, int(round(hop_seconds * fs)))

    if method == "rms":
        frames = _frame_audio(x, frame_len, hop_len)
        if not frames:
            return {
                "method": method,
                "channel_policy": channel_policy,
                "segments": [],
                "total_duration_s": 0.0,
                "longest_duration_s": 0.0,
                "count": 0,
            }
        eps = 1e-12
        rms_db = np.array([20.0 * np.log10(f.value + eps) for f in frames])
        threshold_dbfs = cfg.get("rms_threshold_dbfs")
        if threshold_dbfs is None:
            baseline = float(np.median(rms_db))
            delta = float(cfg.get("rms_delta_db", 12.0))
            threshold_dbfs = baseline + delta
        threshold_linear = 10.0 ** (float(threshold_dbfs) / 20.0)
        segment_summary = _build_segments(
            frames,
            fs=fs,
            frame_len=frame_len,
            total_samples=x.size,
            threshold=threshold_linear,
            min_duration_seconds=min_duration_seconds,
            merge_gap_seconds=merge_gap_seconds,
        )
        return {
            "method": method,
            "channel_policy": channel_policy,
            "threshold": float(threshold_dbfs),
            "threshold_units": "dBFS",
            **segment_summary,
        }

    if method != "spectral_flux":
        raise ValueError("method must be 'spectral_flux' or 'rms'.")

    features = _compute_spectral_flux(x, frame_len, hop_len)
    if not features:
        return {
            "method": method,
            "channel_policy": channel_policy,
            "segments": [],
            "total_duration_s": 0.0,
            "longest_duration_s": 0.0,
            "count": 0,
        }
    flux_values = np.array([f.value for f in features])
    threshold = cfg.get("flux_threshold")
    if threshold is None:
        baseline = float(np.median(flux_values))
        delta = float(cfg.get("flux_delta", 0.2))
        threshold = baseline + delta
    segment_summary = _build_segments(
        features,
        fs=fs,
        frame_len=frame_len,
        total_samples=x.size,
        threshold=float(threshold),
        min_duration_seconds=min_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )
    return {
        "method": method,
        "channel_policy": channel_policy,
        "threshold": float(threshold),
        "threshold_units": "flux",
        **segment_summary,
    }
