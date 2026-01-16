from __future__ import annotations

DEFAULT_SILENCE_GAP_THRESHOLDS = {
    "warn_count": 1,
    "fail_count": 3,
    "warn_total_seconds": 0.2,
    "fail_total_seconds": 0.5,
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


def evaluate_silence_gaps(metrics: dict, *, config: dict | None = None) -> list[dict]:
    """Evaluate silence gap metrics against thresholds and return flags."""
    cfg = build_silence_detection_config(config)
    gaps_cfg = cfg.get("gaps", DEFAULT_SILENCE_GAP_THRESHOLDS)
    gaps = metrics.get("gaps", {})
    if not metrics.get("content_valid", True):
        return []
    count = int(gaps.get("count", 0))
    total_seconds = float(gaps.get("total_duration_s", 0.0))
    status = _status_from_counts(
        count=count,
        total_seconds=total_seconds,
        warn_count=int(gaps_cfg.get("warn_count", 1)),
        fail_count=int(gaps_cfg.get("fail_count", 3)),
        warn_total_seconds=float(gaps_cfg.get("warn_total_seconds", 0.2)),
        fail_total_seconds=float(gaps_cfg.get("fail_total_seconds", 0.5)),
    )
    if not status:
        return []
    return [
        {
            "rule_id": "silence_gap_segments",
            "status": status,
            "measurements": {
                "count": count,
                "total_duration_s": total_seconds,
            },
        }
    ]
