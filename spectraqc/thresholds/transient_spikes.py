from __future__ import annotations

DEFAULT_TRANSIENT_SPIKE_CONFIG = {
    "channel_policy": "per_channel",
    "highpass_hz": 2000.0,
    "derivative_threshold": 0.2,
    "min_separation_seconds": 0.005,
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


def build_transient_spike_config(overrides: dict | None = None) -> dict:
    """Return merged transient spike detection configuration with defaults applied."""
    return _merge_config(DEFAULT_TRANSIENT_SPIKE_CONFIG, overrides)
