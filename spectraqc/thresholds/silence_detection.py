from __future__ import annotations

DEFAULT_SILENCE_GAP_THRESHOLDS = {
    "warn_count": 1,
    "fail_count": 3,
    "warn_total_seconds": 0.2,
    "fail_total_seconds": 0.5,
    "max_gap_seconds": 0.0,
}

DEFAULT_SILENCE_DETECTION_CONFIG = {
    "min_rms_dbfs": -60.0,
    "frame_seconds": 0.1,
    "hop_seconds": 0.1,
    "min_duration_seconds": 0.3,
    "leading_threshold_seconds": 0.2,
    "trailing_threshold_seconds": 0.2,
    "min_content_seconds": 1.0,
    "gaps": DEFAULT_SILENCE_GAP_THRESHOLDS,
}


def _merge_config(base: dict, overrides: dict | None) -> dict:
    if not overrides:
        return base
    merged = {**base}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_config(base[key], value)
        else:
            merged[key] = value
    return merged


def build_silence_detection_config(overrides: dict | None = None) -> dict:
    """Return merged silence detection configuration with defaults applied."""
    return _merge_config(DEFAULT_SILENCE_DETECTION_CONFIG, overrides)


def _status_from_counts(
    *,
    count: int,
    total_seconds: float,
    warn_count: int,
    fail_count: int,
    warn_total_seconds: float,
    fail_total_seconds: float,
) -> str | None:
    if count >= fail_count or total_seconds >= fail_total_seconds:
        return "fail"
    if count >= warn_count or total_seconds >= warn_total_seconds:
        return "warn"
    return None


def summarize_silence_gaps(metrics: dict, *, config: dict | None = None) -> dict:
    """Summarize silence gap metrics with per-gap status details."""
    cfg = build_silence_detection_config(config)
    gaps_cfg = cfg.get("gaps", DEFAULT_SILENCE_GAP_THRESHOLDS)
    gaps = metrics.get("gaps", {})
    max_gap_seconds = float(gaps_cfg.get("max_gap_seconds", 0.0))
    segments = []
    over_max_count = 0
    for seg in gaps.get("segments", []):
        duration = float(seg.get("duration_s", 0.0))
        seg_status = "pass"
        if max_gap_seconds > 0 and duration > max_gap_seconds:
            seg_status = "fail"
            over_max_count += 1
        segments.append({**seg, "status": seg_status})
    count = int(gaps.get("count", len(segments)))
    total_seconds = float(gaps.get("total_duration_s", 0.0))
    longest_gap = float(gaps.get("longest_gap_s", 0.0))
    if not metrics.get("content_valid", True):
        status = "not_applicable"
    else:
        status = _status_from_counts(
            count=count,
            total_seconds=total_seconds,
            warn_count=int(gaps_cfg.get("warn_count", 1)),
            fail_count=int(gaps_cfg.get("fail_count", 3)),
            warn_total_seconds=float(gaps_cfg.get("warn_total_seconds", 0.2)),
            fail_total_seconds=float(gaps_cfg.get("fail_total_seconds", 0.5)),
        )
        if max_gap_seconds > 0 and over_max_count > 0:
            status = "fail"
        if status is None:
            status = "pass"
    return {
        "count": count,
        "total_duration_s": total_seconds,
        "longest_gap_s": longest_gap,
        "max_allowed_gap_s": max_gap_seconds,
        "over_max_count": over_max_count,
        "status": status,
        "segments": segments,
    }


def evaluate_silence_gaps(metrics: dict, *, config: dict | None = None) -> list[dict]:
    """Evaluate silence gap metrics against thresholds and return flags."""
    summary = summarize_silence_gaps(metrics, config=config)
    status = summary.get("status")
    if status not in {"warn", "fail"}:
        return []
    return [
        {
            "rule_id": "silence_gap_segments",
            "status": status,
            "measurements": {
                "count": summary.get("count", 0),
                "total_duration_s": summary.get("total_duration_s", 0.0),
                "over_max_count": summary.get("over_max_count", 0),
                "max_allowed_gap_s": summary.get("max_allowed_gap_s", 0.0),
            },
        }
    ]
