"""Silence detection metrics."""
from __future__ import annotations

import numpy as np


def detect_silence_segments(
    samples: np.ndarray,
    fs: float,
    *,
    min_rms_dbfs: float,
    frame_seconds: float,
    hop_seconds: float,
    min_duration_seconds: float,
) -> dict:
    """Detect silent segments using short-term RMS and duration gating."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("detect_silence_segments expects mono audio.")
    if fs <= 0:
        raise ValueError("detect_silence_segments expects positive sample rate.")

    duration = x.size / fs if x.size else 0.0
    if x.size == 0:
        return {
            "segments": [],
            "total_silence_s": 0.0,
            "silence_ratio": 1.0,
            "longest_silence_s": 0.0,
            "segment_count": 0,
        }

    frame_len = max(1, int(round(frame_seconds * fs)))
    hop_len = max(1, int(round(hop_seconds * fs)))
    threshold = 10.0 ** (min_rms_dbfs / 20.0)
    min_duration_seconds = max(0.0, float(min_duration_seconds))

    starts = []
    silent_flags = []
    for start in range(0, x.size, hop_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        starts.append(start)
        silent_flags.append(rms <= threshold)

    segments: list[dict] = []
    in_run = False
    run_start = 0
    last_start = 0
    for idx, silent in enumerate(silent_flags):
        start = starts[idx]
        if silent and not in_run:
            run_start = start
            in_run = True
        if silent:
            last_start = start
        if in_run and not silent:
            end = min(last_start + frame_len, x.size)
            start_s = run_start / fs
            end_s = end / fs
            duration_s = end_s - start_s
            if duration_s >= min_duration_seconds:
                segments.append(
                    {
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "duration_s": float(duration_s),
                    }
                )
            in_run = False

    if in_run:
        end = min(last_start + frame_len, x.size)
        start_s = run_start / fs
        end_s = end / fs
        duration_s = end_s - start_s
        if duration_s >= min_duration_seconds:
            segments.append(
                {
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(duration_s),
                }
            )

    total_silence = float(sum(seg["duration_s"] for seg in segments))
    silence_ratio = total_silence / duration if duration > 0 else 1.0
    longest = float(max((seg["duration_s"] for seg in segments), default=0.0))

    return {
        "segments": segments,
        "total_silence_s": total_silence,
        "silence_ratio": float(silence_ratio),
        "longest_silence_s": longest,
        "segment_count": int(len(segments)),
    }
