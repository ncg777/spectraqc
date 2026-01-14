"""Algorithm registry for metric and normalization steps."""
from __future__ import annotations


LOUDNESS_ALGO_ID = "bs1770-4-ffmpeg-ebur128-lufs"
TRUE_PEAK_ALGO_ID = "true_peak_kaiser_sinc_4x_v1"


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
        "channel_policy_v1": {
            "id": "channel_policy_v1",
            "params": {
                "policy": str(channel_policy)
            }
        }
    }


def algorithm_ids_from_registry(registry: dict) -> list[str]:
    """Return sorted algorithm IDs from registry."""
    return sorted(registry.keys())
