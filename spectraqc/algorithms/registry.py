"""Algorithm registry for metric and normalization steps."""
from __future__ import annotations


LOUDNESS_ALGO_ID = "bs1770-4-ffmpeg-ebur128-lufs"
TRUE_PEAK_ALGO_ID = "true_peak_kaiser_sinc_4x_v1"
STEREO_CORRELATION_ALGO_ID = "stereo_correlation_pearson_windowed_v1"


def build_algorithm_registry(
    *,
    analysis_lock: dict,
    smoothing_cfg: dict,
    channel_policy: str
) -> dict:
    """Build the algorithm registry with locked parameters."""
    nfft = int(analysis_lock.get("fft_size", 4096))
    hop = int(analysis_lock.get("hop_size", max(1, nfft // 2)))
    window = analysis_lock.get("window", "hann")
    psd_estimator = analysis_lock.get("psd_estimator", "welch")
    dynamic_range_cfg = analysis_lock.get("dynamic_range", {})
    rms_cfg = dynamic_range_cfg.get("rms_percentile", {})
    short_term_cfg = dynamic_range_cfg.get("short_term_lufs", {})

    smoothing_type = smoothing_cfg.get("type", "none")
    smoothing_entry: dict
    if smoothing_type == "octave_fraction":
        smoothing_entry = {
            "id": "smoothing_octave_fraction_v1",
            "params": {
                "type": "octave_fraction",
                "octave_fraction": float(smoothing_cfg.get("octave_fraction", 1 / 6)),
                "min_hz": 20.0
            }
        }
    elif smoothing_type == "log_hz":
        smoothing_entry = {
            "id": "smoothing_log_hz_v1",
            "params": {
                "type": "log_hz",
                "bins_per_octave": int(smoothing_cfg.get("log_hz_bins_per_octave", 12)),
                "min_hz": 20.0
            }
        }
    else:
        smoothing_entry = {
            "id": "smoothing_none_v1",
            "params": {
                "type": "none"
            }
        }

    return {
        "ltpsd_welch_hann_powerhz_v1": {
            "id": "ltpsd_welch_hann_powerhz_v1",
            "params": {
                "fft_size": nfft,
                "hop_size": hop,
                "window": window,
                "psd_estimator": psd_estimator,
                "one_sided": True,
                "power_norm": "power_per_hz"
            }
        },
        "interp_linear_clamped_v1": {
            "id": "interp_linear_clamped_v1",
            "params": {
                "method": "linear",
                "edge": "clamp"
            }
        },
        smoothing_entry["id"]: smoothing_entry,
        "deviation_diff_db_v1": {
            "id": "deviation_diff_db_v1",
            "params": {
                "operation": "input_minus_reference"
            }
        },
        "band_metrics_df_weighted_v1": {
            "id": "band_metrics_df_weighted_v1",
            "params": {
                "weights": "df",
                "variance_ratio_eps": 1e-12
            }
        },
        "spectral_tilt_regress_log2_v1": {
            "id": "spectral_tilt_regress_log2_v1",
            "params": {
                "f_low_hz": 50.0,
                "f_high_hz": 16000.0
            }
        },
        LOUDNESS_ALGO_ID: {
            "id": LOUDNESS_ALGO_ID,
            "params": {
                "standard": "bs1770-4",
                "backend": "ffmpeg_ebur128"
            }
        },
        TRUE_PEAK_ALGO_ID: {
            "id": TRUE_PEAK_ALGO_ID,
            "params": {
                "standard": "kaiser_sinc",
                "backend": "internal",
                "oversample": 4
            }
        },
        STEREO_CORRELATION_ALGO_ID: {
            "id": STEREO_CORRELATION_ALGO_ID,
            "params": {
                "method": "pearson_windowed",
                "frame_seconds": float(
                    analysis_lock.get("stereo_correlation", {}).get("frame_seconds", 0.5)
                ),
                "hop_seconds": float(
                    analysis_lock.get("stereo_correlation", {}).get("hop_seconds", 0.25)
                ),
                "inversion_threshold": float(
                    analysis_lock.get("stereo_correlation", {}).get(
                        "inversion_threshold", -0.8
                    )
                ),
            },
        },
        "channel_policy_v1": {
            "id": "channel_policy_v1",
            "params": {
                "policy": str(channel_policy)
            }
        },
        "dynamic_range_rms_percentile_v1": {
            "id": "dynamic_range_rms_percentile_v1",
            "params": {
                "frame_seconds": float(rms_cfg.get("frame_seconds", 3.0)),
                "hop_seconds": float(rms_cfg.get("hop_seconds", 1.0)),
                "low_percentile": float(rms_cfg.get("low_percentile", 10.0)),
                "high_percentile": float(rms_cfg.get("high_percentile", 95.0)),
            },
        },
        "dynamic_range_short_term_lufs_v1": {
            "id": "dynamic_range_short_term_lufs_v1",
            "params": {
                "low_percentile": float(short_term_cfg.get("low_percentile", 10.0)),
                "high_percentile": float(short_term_cfg.get("high_percentile", 95.0)),
            },
        }
    }


def algorithm_ids_from_registry(registry: dict) -> list[str]:
    """Return sorted algorithm IDs from registry."""
    return sorted(registry.keys())
