from __future__ import annotations

import numpy as np


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("Expected non-empty audio array.")
    return samples


def _crest_factor_metrics_mono(
    x: np.ndarray,
    *,
    fs: float,
    frame_seconds: float,
    hop_seconds: float,
    min_crest_db: float,
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    frame_len = max(1, int(round(frame_seconds * fs)))
    hop_len = max(1, int(round(hop_seconds * fs)))
    if x.size < frame_len:
        frame_len = x.size
        hop_len = x.size
    count = 0
    max_crest = None
    for start in range(0, x.size - frame_len + 1, hop_len):
        frame = x[start:start + frame_len]
        peak = float(np.max(np.abs(frame)))
        if peak <= 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms <= 0:
            continue
        crest = float(20.0 * np.log10(peak / rms))
        if crest >= min_crest_db:
            count += 1
            if max_crest is None or crest > max_crest:
                max_crest = crest
    total_duration_s = float(count * hop_seconds)
    return {
        "frame_seconds": float(frame_seconds),
        "hop_seconds": float(hop_seconds),
        "min_crest_db": float(min_crest_db),
        "count": int(count),
        "total_duration_s": total_duration_s,
        "max_crest_db": float(max_crest) if max_crest is not None else None,
    }


def _near_zero_peak_metrics_mono(
    x: np.ndarray,
    *,
    fs: float,
    threshold_dbfs: float,
    min_separation_seconds: float,
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    threshold_amp = float(10.0 ** (threshold_dbfs / 20.0))
    min_sep_samples = max(1, int(round(min_separation_seconds * fs)))
    indices = np.flatnonzero(np.abs(x) >= threshold_amp)
    count = 0
    if indices.size:
        last_idx = indices[0]
        count = 1
        for idx in indices[1:]:
            if idx - last_idx >= min_sep_samples:
                count += 1
                last_idx = idx
    duration_s = float(x.size / fs) if fs > 0 else 0.0
    rate_per_s = float(count / duration_s) if duration_s > 0 else 0.0
    return {
        "threshold_dbfs": float(threshold_dbfs),
        "min_separation_seconds": float(min_separation_seconds),
        "count": int(count),
        "rate_per_s": rate_per_s,
    }


def detect_peak_anomalies(
    samples: np.ndarray,
    fs: float,
    *,
    config: dict | None = None,
) -> dict:
    """Detect high crest factor anomalies and near-0 dBFS peaks."""
    samples = _validate_samples(samples)
    cfg = config or {}
    crest_cfg = cfg.get("crest_factor", {})
    peak_cfg = cfg.get("near_zero_peaks", {})
    policy = str(cfg.get("channel_policy", "per_channel")).lower()

    crest_kwargs = {
        "fs": float(fs),
        "frame_seconds": float(crest_cfg.get("frame_seconds", 0.05)),
        "hop_seconds": float(crest_cfg.get("hop_seconds", 0.025)),
        "min_crest_db": float(crest_cfg.get("min_crest_db", 20.0)),
    }
    peak_kwargs = {
        "fs": float(fs),
        "threshold_dbfs": float(peak_cfg.get("threshold_dbfs", -0.2)),
        "min_separation_seconds": float(peak_cfg.get("min_separation_seconds", 0.01)),
    }

    if samples.ndim == 1:
        return {
            "policy": "mono",
            "crest_factor": _crest_factor_metrics_mono(samples, **crest_kwargs),
            "near_zero_peaks": _near_zero_peak_metrics_mono(samples, **peak_kwargs),
        }

    if samples.ndim != 2:
        raise ValueError("Expected mono or stereo audio array.")

    channel_reports = []
    for idx in range(samples.shape[1]):
        channel_reports.append(
            {
                "channel_index": int(idx),
                "crest_factor": _crest_factor_metrics_mono(samples[:, idx], **crest_kwargs),
                "near_zero_peaks": _near_zero_peak_metrics_mono(samples[:, idx], **peak_kwargs),
            }
        )

    if policy == "average":
        crest_counts = [c["crest_factor"]["count"] for c in channel_reports]
        crest_duration = [c["crest_factor"]["total_duration_s"] for c in channel_reports]
        crest_max = [
            c["crest_factor"]["max_crest_db"]
            for c in channel_reports
            if c["crest_factor"]["max_crest_db"] is not None
        ]
        peak_counts = [c["near_zero_peaks"]["count"] for c in channel_reports]
        peak_rates = [c["near_zero_peaks"]["rate_per_s"] for c in channel_reports]
        return {
            "policy": "average",
            "crest_factor": {
                **channel_reports[0]["crest_factor"],
                "count": int(np.mean(crest_counts)),
                "total_duration_s": float(np.mean(crest_duration)),
                "max_crest_db": float(np.max(crest_max)) if crest_max else None,
            },
            "near_zero_peaks": {
                **channel_reports[0]["near_zero_peaks"],
                "count": int(np.mean(peak_counts)),
                "rate_per_s": float(np.mean(peak_rates)),
            },
        }

    return {
        "policy": "per_channel",
        "channels": channel_reports,
    }
