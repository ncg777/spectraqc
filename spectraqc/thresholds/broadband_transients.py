from __future__ import annotations

DEFAULT_BROADBAND_TRANSIENT_THRESHOLDS = {
    "warn_count": 1,
    "fail_count": 3,
    "warn_total_seconds": 0.1,
    "fail_total_seconds": 0.3,
}

DEFAULT_BROADBAND_TRANSIENT_CONFIG = {
    "method": "spectral_flux",
    "channel_policy": "average",
    "frame_seconds": 0.05,
    "hop_seconds": 0.02,
    "min_duration_seconds": 0.02,
    "merge_gap_seconds": 0.02,
    "flux_delta": 0.2,
    "rms_delta_db": 12.0,
    "gates": DEFAULT_BROADBAND_TRANSIENT_THRESHOLDS,
}


def _merge_config(base: dict, overrides: dict | None) -> dict:
    if not overrides:
        return dict(base)
    merged = {**base}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_config(base[key], value)
        else:
            merged[key] = value
    return merged


def build_broadband_transient_config(overrides: dict | None = None) -> dict:
    """Return merged broadband transient configuration with defaults applied."""
    return _merge_config(DEFAULT_BROADBAND_TRANSIENT_CONFIG, overrides)


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


def summarize_broadband_transients(metrics: dict, *, config: dict | None = None) -> dict:
    cfg = build_broadband_transient_config(config)
    gates_cfg = cfg.get("gates", DEFAULT_BROADBAND_TRANSIENT_THRESHOLDS)
    count = int(metrics.get("count", 0))
    total_seconds = float(metrics.get("total_duration_s", 0.0))
    longest = float(metrics.get("longest_duration_s", 0.0))
    status = _status_from_counts(
        count=count,
        total_seconds=total_seconds,
        warn_count=int(gates_cfg.get("warn_count", 1)),
        fail_count=int(gates_cfg.get("fail_count", 3)),
        warn_total_seconds=float(gates_cfg.get("warn_total_seconds", 0.1)),
        fail_total_seconds=float(gates_cfg.get("fail_total_seconds", 0.3)),
    )
    if status is None:
        status = "pass"
    return {
        "count": count,
        "total_duration_s": total_seconds,
        "longest_duration_s": longest,
        "status": status,
        "segments": metrics.get("segments", []),
    }


def evaluate_broadband_transients(metrics: dict, *, config: dict | None = None) -> list[dict]:
    summary = summarize_broadband_transients(metrics, config=config)
    status = summary.get("status")
    if status not in {"warn", "fail"}:
        return []
    cfg = build_broadband_transient_config(config)
    gates_cfg = cfg.get("gates", DEFAULT_BROADBAND_TRANSIENT_THRESHOLDS)
    return [
        {
            "rule_id": "broadband_transient_segments",
            "status": status,
            "measurements": {
                "count": summary.get("count", 0),
                "total_duration_s": summary.get("total_duration_s", 0.0),
                "longest_duration_s": summary.get("longest_duration_s", 0.0),
                "warn_count": int(gates_cfg.get("warn_count", 1)),
                "fail_count": int(gates_cfg.get("fail_count", 3)),
                "warn_total_seconds": float(gates_cfg.get("warn_total_seconds", 0.1)),
                "fail_total_seconds": float(gates_cfg.get("fail_total_seconds", 0.3)),
            },
        }
    ]
