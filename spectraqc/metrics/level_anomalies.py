"""Level anomaly detectors for sudden drops or zeroed segments."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class _Segment:
    start_s: float
    end_s: float
    duration_s: float
    metadata: dict


def _frame_rms_dbfs(
    x: np.ndarray,
    fs: float,
    *,
    frame_seconds: float,
    hop_seconds: float,
    floor_dbfs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fs <= 0:
        raise ValueError("Sample rate must be positive.")
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive.")
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive.")
    frame_len = max(1, int(round(frame_seconds * fs)))
    hop_len = max(1, int(round(hop_seconds * fs)))
    if x.size < frame_len:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.arange(0, x.size - frame_len + 1, hop_len, dtype=np.int64)
    ends = starts + frame_len
    rms_vals = []
    for start, end in zip(starts, ends):
        frame = x[start:end]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms <= 0:
            rms_dbfs = float(floor_dbfs)
        else:
            rms_dbfs = float(max(20.0 * np.log10(rms), floor_dbfs))
        rms_vals.append(rms_dbfs)
    return np.array(rms_vals, dtype=np.float64), starts, ends


def _drop_segments_mono(
    x: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.1,
    hop_seconds: float = 0.05,
    baseline_window_seconds: float = 1.0,
    drop_db: float = 24.0,
    min_duration_seconds: float = 0.1,
    floor_dbfs: float = -120.0,
) -> list[_Segment]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        return []

    rms_dbfs, starts, ends = _frame_rms_dbfs(
        x,
        fs,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        floor_dbfs=floor_dbfs,
    )
    if rms_dbfs.size == 0:
        return []

    window_frames = max(1, int(round(baseline_window_seconds / hop_seconds)))
    baselines = np.full_like(rms_dbfs, np.nan, dtype=np.float64)
    for i in range(rms_dbfs.size):
        start = max(0, i - window_frames)
        if start >= i:
            continue
        window = rms_dbfs[start:i]
        if window.size == 0:
            continue
        baselines[i] = float(np.median(window))

    valid = np.isfinite(baselines)
    mask = np.zeros_like(rms_dbfs, dtype=bool)
    mask[valid] = (baselines[valid] - rms_dbfs[valid]) >= float(drop_db)

    segments = []
    if np.any(mask):
        idx = np.flatnonzero(mask)
        gap = np.diff(idx)
        run_starts = np.concatenate(([0], np.where(gap > 1)[0] + 1))
        run_ends = np.concatenate((np.where(gap > 1)[0], [len(idx) - 1]))
        for rs, re in zip(run_starts, run_ends):
            start_idx = idx[rs]
            end_idx = idx[re]
            start_s = float(starts[start_idx] / fs)
            end_s = float(ends[end_idx] / fs)
            duration = end_s - start_s
            if duration < min_duration_seconds:
                continue
            segment_baseline = float(np.median(baselines[idx[rs:re + 1]]))
            segment_level = float(np.median(rms_dbfs[idx[rs:re + 1]]))
            metadata = {
                "baseline_dbfs": segment_baseline,
                "level_dbfs": segment_level,
                "drop_db": float(segment_baseline - segment_level),
            }
            segments.append(
                _Segment(
                    start_s=start_s,
                    end_s=end_s,
                    duration_s=duration,
                    metadata=metadata,
                )
            )
    return segments


def _zero_segments_mono(
    x: np.ndarray,
    fs: float,
    *,
    zero_threshold: float = 1e-8,
    min_duration_seconds: float = 0.01,
) -> list[_Segment]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected 1D mono audio array.")
    if x.size == 0:
        return []
    if zero_threshold < 0:
        raise ValueError("zero_threshold must be non-negative.")

    mask = np.abs(x) <= float(zero_threshold)
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    gap = np.diff(idx)
    run_starts = np.concatenate(([0], np.where(gap > 1)[0] + 1))
    run_ends = np.concatenate((np.where(gap > 1)[0], [len(idx) - 1]))
    segments: list[_Segment] = []
    for rs, re in zip(run_starts, run_ends):
        start_idx = int(idx[rs])
        end_idx = int(idx[re]) + 1
        start_s = float(start_idx / fs)
        end_s = float(end_idx / fs)
        duration = end_s - start_s
        if duration < min_duration_seconds:
            continue
        segments.append(
            _Segment(
                start_s=start_s,
                end_s=end_s,
                duration_s=duration,
                metadata={},
            )
        )
    return segments


def _summarize_segments(segments: list[_Segment]) -> dict:
    return {
        "count": int(len(segments)),
        "total_duration_s": float(sum(seg.duration_s for seg in segments)),
        "segments": [
            {
                "start_s": seg.start_s,
                "end_s": seg.end_s,
                "duration_s": seg.duration_s,
                **seg.metadata,
            }
            for seg in segments
        ],
    }


def detect_level_anomalies(
    samples: np.ndarray,
    fs: float,
    *,
    config: dict | None = None,
) -> dict:
    """
    Detect sudden level drops and full-scale zero segments.

    Args:
        samples: Mono or stereo audio samples.
        fs: Sample rate.
        config: Configuration dict with "channel_policy", "drop", and "zero" blocks.

    Returns:
        Dictionary with policy, per-channel details, and summary counts.
    """
    cfg = config or {}
    policy = str(cfg.get("channel_policy", "per_channel")).strip().lower()
    drop_cfg = cfg.get("drop", {})
    zero_cfg = cfg.get("zero", {})
    drop_params = {
        key: drop_cfg[key]
        for key in (
            "frame_seconds",
            "hop_seconds",
            "baseline_window_seconds",
            "drop_db",
            "min_duration_seconds",
            "floor_dbfs",
        )
        if key in drop_cfg
    }
    zero_params = {
        key: zero_cfg[key]
        for key in (
            "zero_threshold",
            "min_duration_seconds",
        )
        if key in zero_cfg
    }

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        drop_segments = _drop_segments_mono(samples, fs, **drop_params)
        zero_segments = _zero_segments_mono(samples, fs, **zero_params)
        return {
            "policy": "mono",
            "drop": _summarize_segments(drop_segments),
            "zero": _summarize_segments(zero_segments),
        }

    if samples.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")

    if policy == "average":
        mono = np.mean(samples, axis=1)
        drop_segments = _drop_segments_mono(mono, fs, **drop_params)
        zero_segments = _zero_segments_mono(mono, fs, **zero_params)
        return {
            "policy": "average",
            "drop": _summarize_segments(drop_segments),
            "zero": _summarize_segments(zero_segments),
        }

    if policy != "per_channel":
        raise ValueError("channel_policy must be 'per_channel' or 'average'.")

    channel_reports = []
    total_drop_segments: list[_Segment] = []
    total_zero_segments: list[_Segment] = []
    for idx in range(samples.shape[1]):
        channel = samples[:, idx]
        drop_segments = _drop_segments_mono(channel, fs, **drop_params)
        zero_segments = _zero_segments_mono(channel, fs, **zero_params)
        channel_reports.append(
            {
                "channel_index": int(idx),
                "drop": _summarize_segments(drop_segments),
                "zero": _summarize_segments(zero_segments),
            }
        )
        total_drop_segments.extend(drop_segments)
        total_zero_segments.extend(zero_segments)

    return {
        "policy": "per_channel",
        "drop": _summarize_segments(total_drop_segments),
        "zero": _summarize_segments(total_zero_segments),
        "channels": channel_reports,
    }
