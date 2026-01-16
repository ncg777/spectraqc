from __future__ import annotations


DEFAULT_INTER_CHANNEL_DELAY_THRESHOLDS = {
    "max_delay_seconds": 0.01,
    "pass": 0.0005,
    "warn": 0.001,
}


def build_inter_channel_delay_config(overrides: dict | None = None) -> dict:
    """Return merged inter-channel delay thresholds with defaults applied."""
    if not overrides:
        return dict(DEFAULT_INTER_CHANNEL_DELAY_THRESHOLDS)
    cfg = dict(DEFAULT_INTER_CHANNEL_DELAY_THRESHOLDS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
