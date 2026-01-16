from __future__ import annotations


DEFAULT_LEVEL_METRIC_THRESHOLDS = {
    "peak_dbfs": {"pass": -1.0, "warn": -0.3},
    "true_peak_dbtp": {"pass": -1.0, "warn": -0.5},
    "rms_dbfs": {"pass": -20.0, "warn": -14.0},
    "lufs_i": {"pass": -16.0, "warn": -14.0},
    "clipped_samples": {"pass": 0.0, "warn": 10.0},
    "clipped_runs": {"pass": 0.0, "warn": 1.0},
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


def build_level_metric_thresholds(overrides: dict | None = None) -> dict:
    """Return merged level metric thresholds with defaults applied."""
    return _merge_config(DEFAULT_LEVEL_METRIC_THRESHOLDS, overrides)
