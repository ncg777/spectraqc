from __future__ import annotations

DEFAULT_LEVEL_ANOMALY_THRESHOLDS = {
    "channel_policy": "per_channel",
    "drop": {
        "frame_seconds": 0.1,
        "hop_seconds": 0.05,
        "baseline_window_seconds": 1.0,
        "drop_db": 24.0,
        "min_duration_seconds": 0.1,
        "floor_dbfs": -120.0,
        "warn_count": 1,
        "fail_count": 3,
        "warn_total_seconds": 0.1,
        "fail_total_seconds": 0.5,
    },
    "zero": {
        "zero_threshold": 1e-8,
        "min_duration_seconds": 0.01,
        "warn_count": 1,
        "fail_count": 3,
        "warn_total_seconds": 0.1,
        "fail_total_seconds": 0.5,
    },
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


def build_level_anomaly_config(overrides: dict | None = None) -> dict:
    """Return merged level anomaly configuration with defaults applied."""
    return _merge_config(DEFAULT_LEVEL_ANOMALY_THRESHOLDS, overrides)


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


def _evaluate_segment_block(
    *,
    block: dict,
    thresholds: dict,
    rule_id: str,
    channel_index: int | None = None,
) -> dict | None:
    count = int(block.get("count", 0))
    total_seconds = float(block.get("total_duration_s", 0.0))
    status = _status_from_counts(
        count=count,
        total_seconds=total_seconds,
        warn_count=int(thresholds.get("warn_count", 1)),
        fail_count=int(thresholds.get("fail_count", 3)),
        warn_total_seconds=float(thresholds.get("warn_total_seconds", 0.1)),
        fail_total_seconds=float(thresholds.get("fail_total_seconds", 0.5)),
    )
    if not status:
        return None
    measurements = {
        "count": count,
        "total_duration_s": total_seconds,
    }
    if channel_index is not None:
        measurements["channel_index"] = int(channel_index)
    return {
        "rule_id": rule_id,
        "status": status,
        "measurements": measurements,
    }


def evaluate_level_anomalies(metrics: dict, *, config: dict | None = None) -> list[dict]:
    """Evaluate level anomaly metrics and return rule flags."""
    cfg = build_level_anomaly_config(config)
    flags: list[dict] = []
    policy = str(metrics.get("policy", cfg.get("channel_policy", "per_channel"))).lower()
    drop_thresholds = cfg.get("drop", {})
    zero_thresholds = cfg.get("zero", {})

    if policy == "per_channel" and metrics.get("channels"):
        for channel in metrics.get("channels", []):
            idx = channel.get("channel_index")
            drop_flag = _evaluate_segment_block(
                block=channel.get("drop", {}),
                thresholds=drop_thresholds,
                rule_id="level_drop_segments",
                channel_index=idx,
            )
            if drop_flag:
                flags.append(drop_flag)
            zero_flag = _evaluate_segment_block(
                block=channel.get("zero", {}),
                thresholds=zero_thresholds,
                rule_id="zero_level_segments",
                channel_index=idx,
            )
            if zero_flag:
                flags.append(zero_flag)
    else:
        drop_flag = _evaluate_segment_block(
            block=metrics.get("drop", {}),
            thresholds=drop_thresholds,
            rule_id="level_drop_segments",
        )
        if drop_flag:
            flags.append(drop_flag)
        zero_flag = _evaluate_segment_block(
            block=metrics.get("zero", {}),
            thresholds=zero_thresholds,
            rule_id="zero_level_segments",
        )
        if zero_flag:
            flags.append(zero_flag)

    return flags
