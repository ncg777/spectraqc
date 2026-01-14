from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_profile_dict(
    *,
    smoothing: dict | None = None,
    true_peak_enabled: bool = False
) -> dict:
    freqs = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
    smoothing_cfg = smoothing or {"type": "none"}
    normalization = {
        "loudness": {"enabled": False, "target_lufs_i": -14.0, "algorithm_id": ""},
        "true_peak": {"enabled": bool(true_peak_enabled), "max_dbtp": -1.0, "algorithm_id": "tp"}
    }
    return {
        "profile": {"name": "test_profile", "kind": "streaming", "version": "2.0"},
        "frequency_grid": {"freqs_hz": freqs},
        "reference_curves": {
            "mean_db": [-60.0] * len(freqs),
            "var_db2": [1.0] * len(freqs)
        },
        "bands": [
            {"name": "low", "f_low_hz": 20, "f_high_hz": 200},
            {"name": "mid", "f_low_hz": 200, "f_high_hz": 2000},
            {"name": "high", "f_low_hz": 2000, "f_high_hz": 20000}
        ],
        "analysis_lock": {
            "fft_size": 1024,
            "hop_size": 512,
            "window": "hann",
            "psd_estimator": "welch",
            "channel_policy": "mono",
            "smoothing": smoothing_cfg,
            "normalization": normalization,
        },
        "threshold_model": {
            "rules": {
                "band_mean": {
                    "default": {"pass": 1.0, "warn": 2.0},
                    "by_band": []
                },
                "band_max": {
                    "default": {"pass": 3.0, "warn": 6.0}
                },
                "tilt": {"pass": 0.5, "warn": 1.0}
            },
            "aggregation": {"warn_if_warn_band_count_at_least": 2}
        },
        "integrity": {
            "profile_hash_sha256": "0" * 64,
            "signed": False,
            "signature": {"algo": "none", "value_b64": ""}
        }
    }


def write_profile(tmp_path: Path, profile: dict) -> Path:
    path = tmp_path / "profile.ref.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path
