from __future__ import annotations


DEFAULT_PEAK_ANOMALY_THRESHOLDS = {
    "channel_policy": "per_channel",
    "crest_factor": {
        "frame_seconds": 0.05,
        "hop_seconds": 0.025,
        "min_crest_db": 20.0,
        "warn_count": 3,
        "fail_count": 10,
        "warn_total_seconds": 0.1,
        "fail_total_seconds": 0.5,
    },
    "near_zero_peaks": {
        "threshold_dbfs": -0.2,
        "min_separation_seconds": 0.01,
        "warn_rate_per_s": 2.0,
        "fail_rate_per_s": 5.0,
        "warn_count": 5,
        "fail_count": 20,
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


def build_peak_anomaly_config(overrides: dict | None = None) -> dict:
    """Return merged peak anomaly configuration with defaults applied."""
    return _merge_config(DEFAULT_PEAK_ANOMALY_THRESHOLDS, overrides)


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


def _status_from_rate(
    *,
    count: int,
    rate_per_s: float,
    warn_count: int,
    fail_count: int,
    warn_rate: float,
    fail_rate: float,
) -> str | None:
    if count >= fail_count or rate_per_s >= fail_rate:
        return "fail"
    if count >= warn_count or rate_per_s >= warn_rate:
        return "warn"
    return None


def _evaluate_crest_factor(
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
        warn_count=int(thresholds.get("warn_count", 3)),
        fail_count=int(thresholds.get("fail_count", 10)),
        warn_total_seconds=float(thresholds.get("warn_total_seconds", 0.1)),
        fail_total_seconds=float(thresholds.get("fail_total_seconds", 0.5)),
    )
    if not status:
        return None
    measurements = {
        "count": count,
        "total_duration_s": total_seconds,
        "max_crest_db": block.get("max_crest_db"),
        "min_crest_db": block.get("min_crest_db"),
        "frame_seconds": block.get("frame_seconds"),
        "hop_seconds": block.get("hop_seconds"),
    }
    if channel_index is not None:
        measurements["channel_index"] = int(channel_index)
    return {
        "rule_id": rule_id,
        "status": status,
        "measurements": measurements,
    }


def _evaluate_near_zero_peaks(
    *,
    block: dict,
    thresholds: dict,
    rule_id: str,
    channel_index: int | None = None,
) -> dict | None:
    count = int(block.get("count", 0))
    rate = float(block.get("rate_per_s", 0.0))
    status = _status_from_rate(
        count=count,
        rate_per_s=rate,
        warn_count=int(thresholds.get("warn_count", 5)),
        fail_count=int(thresholds.get("fail_count", 20)),
        warn_rate=float(thresholds.get("warn_rate_per_s", 2.0)),
        fail_rate=float(thresholds.get("fail_rate_per_s", 5.0)),
    )
    if not status:
        return None
    measurements = {
        "count": count,
        "rate_per_s": rate,
        "threshold_dbfs": block.get("threshold_dbfs"),
        "min_separation_seconds": block.get("min_separation_seconds"),
    }
    if channel_index is not None:
        measurements["channel_index"] = int(channel_index)
    return {
        "rule_id": rule_id,
        "status": status,
        "measurements": measurements,
    }


def evaluate_peak_anomalies(metrics: dict, *, config: dict | None = None) -> list[dict]:
    """Evaluate peak anomaly metrics and return rule flags."""
    cfg = build_peak_anomaly_config(config)
    policy = str(metrics.get("policy", cfg.get("channel_policy", "per_channel"))).lower()
    crest_cfg = cfg.get("crest_factor", {})
    peak_cfg = cfg.get("near_zero_peaks", {})
    flags: list[dict] = []

    if policy == "per_channel" and metrics.get("channels"):
        for channel in metrics.get("channels", []):
            idx = channel.get("channel_index")
            crest_flag = _evaluate_crest_factor(
                block=channel.get("crest_factor", {}),
                thresholds=crest_cfg,
                rule_id="high_crest_factor_anomalies",
                channel_index=idx,
            )
            if crest_flag:
                flags.append(crest_flag)
            peak_flag = _evaluate_near_zero_peaks(
                block=channel.get("near_zero_peaks", {}),
                thresholds=peak_cfg,
                rule_id="near_zero_dbfs_peaks",
                channel_index=idx,
            )
            if peak_flag:
                flags.append(peak_flag)
    else:
        crest_flag = _evaluate_crest_factor(
            block=metrics.get("crest_factor", {}),
            thresholds=crest_cfg,
            rule_id="high_crest_factor_anomalies",
        )
        if crest_flag:
            flags.append(crest_flag)
        peak_flag = _evaluate_near_zero_peaks(
            block=metrics.get("near_zero_peaks", {}),
            thresholds=peak_cfg,
            rule_id="near_zero_dbfs_peaks",
        )
        if peak_flag:
            flags.append(peak_flag)

    return flags
