"""Channel consistency thresholds."""
from __future__ import annotations


DEFAULT_CHANNEL_CONSISTENCY = {
    "declared": "stereo",
    "frame_seconds": 0.5,
    "hop_seconds": 0.25,
    "mono": {
        "corr_min": {"pass": 0.995, "warn": 0.99},
        "side_max_db": {"pass": -60.0, "warn": -50.0},
    },
    "stereo": {
        "corr": {
            "pass_min": -0.2,
            "pass_max": 0.98,
            "warn_min": -0.4,
            "warn_max": 1.0,
        },
        "side_min_db": {"pass": -30.0, "warn": -40.0},
    },
}


def build_channel_consistency_config(overrides: dict | None = None) -> dict:
    """Return merged channel consistency thresholds with defaults applied."""
    overrides = overrides or {}
    mono_cfg = overrides.get("mono", {})
    stereo_cfg = overrides.get("stereo", {})
    return {
        "declared": str(overrides.get("declared", DEFAULT_CHANNEL_CONSISTENCY["declared"])).lower(),
        "frame_seconds": float(
            overrides.get("frame_seconds", DEFAULT_CHANNEL_CONSISTENCY["frame_seconds"])
        ),
        "hop_seconds": float(
            overrides.get("hop_seconds", DEFAULT_CHANNEL_CONSISTENCY["hop_seconds"])
        ),
        "mono": {
            "corr_min": {
                "pass": float(
                    mono_cfg.get(
                        "corr_min", {}
                    ).get("pass", DEFAULT_CHANNEL_CONSISTENCY["mono"]["corr_min"]["pass"])
                ),
                "warn": float(
                    mono_cfg.get(
                        "corr_min", {}
                    ).get("warn", DEFAULT_CHANNEL_CONSISTENCY["mono"]["corr_min"]["warn"])
                ),
            },
            "side_max_db": {
                "pass": float(
                    mono_cfg.get(
                        "side_max_db", {}
                    ).get("pass", DEFAULT_CHANNEL_CONSISTENCY["mono"]["side_max_db"]["pass"])
                ),
                "warn": float(
                    mono_cfg.get(
                        "side_max_db", {}
                    ).get("warn", DEFAULT_CHANNEL_CONSISTENCY["mono"]["side_max_db"]["warn"])
                ),
            },
        },
        "stereo": {
            "corr": {
                "pass_min": float(
                    stereo_cfg.get(
                        "corr", {}
                    ).get("pass_min", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["corr"]["pass_min"])
                ),
                "pass_max": float(
                    stereo_cfg.get(
                        "corr", {}
                    ).get("pass_max", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["corr"]["pass_max"])
                ),
                "warn_min": float(
                    stereo_cfg.get(
                        "corr", {}
                    ).get("warn_min", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["corr"]["warn_min"])
                ),
                "warn_max": float(
                    stereo_cfg.get(
                        "corr", {}
                    ).get("warn_max", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["corr"]["warn_max"])
                ),
            },
            "side_min_db": {
                "pass": float(
                    stereo_cfg.get(
                        "side_min_db", {}
                    ).get("pass", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["side_min_db"]["pass"])
                ),
                "warn": float(
                    stereo_cfg.get(
                        "side_min_db", {}
                    ).get("warn", DEFAULT_CHANNEL_CONSISTENCY["stereo"]["side_min_db"]["warn"])
                ),
            },
        },
    }
