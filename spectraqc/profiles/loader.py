from __future__ import annotations
import json
import numpy as np
from spectraqc.types import ReferenceProfile, FrequencyBand
from spectraqc.thresholds.brickwall import build_spectral_artifact_config
from spectraqc.thresholds.level_metrics import DEFAULT_LEVEL_METRIC_THRESHOLDS
from spectraqc.thresholds.level_anomalies import build_level_anomaly_config
from spectraqc.thresholds.peak_anomalies import build_peak_anomaly_config
from spectraqc.thresholds.transient_spikes import build_transient_spike_config
from spectraqc.thresholds.broadband_transients import build_broadband_transient_config
from spectraqc.thresholds.inter_channel_delay import build_inter_channel_delay_config
from spectraqc.thresholds.channel_consistency import build_channel_consistency_config
from spectraqc.profiles.validator import validate_reference_profile_dict
from spectraqc.metrics.tonal import derive_noise_floor_baselines
from spectraqc.thresholds.silence_detection import build_silence_detection_config


def load_reference_profile(path: str) -> ReferenceProfile:
    """
    Load a reference profile from JSON file.
    
    Args:
        path: Path to the reference profile JSON file
        
    Returns:
        ReferenceProfile object with all configuration and thresholds
    """
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    validate_reference_profile_dict(j)

    bands = [
        FrequencyBand(b["name"], float(b["f_low_hz"]), float(b["f_high_hz"]))
        for b in j["bands"]
    ]
    freqs = np.array(j["frequency_grid"]["freqs_hz"], dtype=np.float64)
    ref_mean = np.array(j["reference_curves"]["mean_db"], dtype=np.float64)
    ref_var = np.array(
        j["reference_curves"].get("var_db2", [1.0] * len(ref_mean)),
        dtype=np.float64
    )

    analysis_lock = j.get("analysis_lock", {})
    normalization = analysis_lock.get("normalization", {
        "loudness": {"enabled": False, "target_lufs_i": -14.0, "algorithm_id": ""},
        "true_peak": {"enabled": False, "max_dbtp": -1.0, "algorithm_id": ""}
    })

    tm = j["threshold_model"]["rules"]
    corr_rules = tm.get("stereo_correlation", {})
    delay_rules = tm.get("inter_channel_delay")
    consistency_rules = tm.get("channel_consistency")

    def _range_cfg(cfg: dict, *, default_min: float, default_max: float) -> dict:
        return {
            "pass_min": float(cfg.get("pass_min", default_min)),
            "pass_max": float(cfg.get("pass_max", default_max)),
            "warn_min": float(cfg.get("warn_min", default_min)),
            "warn_max": float(cfg.get("warn_max", default_max)),
        }

    # Band mean thresholds
    band_mean_default = (
        float(tm["band_mean"]["default"]["pass"]),
        float(tm["band_mean"]["default"]["warn"])
    )
    band_mean_map = {"default": band_mean_default}
    for bb in tm["band_mean"]["by_band"]:
        band_mean_map[bb["band_name"]] = (float(bb["pass"]), float(bb["warn"]))

    # Band max thresholds
    band_max_default = (
        float(tm["band_max"]["default"]["pass"]),
        float(tm["band_max"]["default"]["warn"])
    )
    band_max_map = {"all": band_max_default}
    for bb in tm["band_max"].get("by_band", []):
        band_max_map[bb["band_name"]] = (float(bb["pass"]), float(bb["warn"]))

    # Tilt threshold
    tilt = (float(tm["tilt"]["pass"]), float(tm["tilt"]["warn"]))

    thresholds = {
        "band_mean_db": band_mean_map,
        "band_max_db": band_max_map,
        "tilt_db_per_oct": tilt,
        "variance_ratio": (1.2, 1.5),
        "warn_if_warn_band_count_at_least": int(
            j["threshold_model"]["aggregation"]["warn_if_warn_band_count_at_least"]
        ),
        "_ref_var_db2": ref_var,
        "_smoothing": analysis_lock.get("smoothing", {"type": "none"}),
        "spectral_artifacts": build_spectral_artifact_config(
            tm.get("spectral_artifacts")
        ),
        "level_anomalies": build_level_anomaly_config(
            tm.get("level_anomalies")
        ),
        "silence_detection": build_silence_detection_config(
            tm.get("silence_detection")
        ),
        "transient_spikes": build_transient_spike_config(
            tm.get("transient_spikes")
        ),
        "broadband_transients": build_broadband_transient_config(
            tm.get("broadband_transients")
        ),
        "peak_anomalies": build_peak_anomaly_config(
            tm.get("peak_anomalies")
        ),
    }

    peak_rules = tm.get("peak_dbfs") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get("peak_dbfs")
    if peak_rules:
        thresholds["peak_dbfs"] = (
            float(peak_rules["pass"]),
            float(peak_rules["warn"]),
        )

    rms_rules = tm.get("rms_dbfs") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get("rms_dbfs")
    if rms_rules:
        thresholds["rms_dbfs"] = (
            float(rms_rules["pass"]),
            float(rms_rules["warn"]),
        )

    rms_balance_rules = tm.get("rms_balance_db") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get(
        "rms_balance_db"
    )
    if rms_balance_rules:
        thresholds["rms_balance_db"] = (
            float(rms_balance_rules["pass"]),
            float(rms_balance_rules["warn"]),
        )

    lufs_rules = tm.get("lufs_i") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get("lufs_i")
    if lufs_rules:
        thresholds["lufs_i"] = (
            float(lufs_rules["pass"]),
            float(lufs_rules["warn"]),
        )

    lufs_balance_rules = tm.get("lufs_balance_lu") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get(
        "lufs_balance_lu"
    )
    if lufs_balance_rules:
        thresholds["lufs_balance_lu"] = (
            float(lufs_balance_rules["pass"]),
            float(lufs_balance_rules["warn"]),
        )

    noise_floor_rules = tm.get("noise_floor_dbfs")
    if noise_floor_rules:
        thresholds["noise_floor_dbfs"] = (
            float(noise_floor_rules["pass"]),
            float(noise_floor_rules["warn"]),
        )

    crest_rules = tm.get("crest_factor_db")
    if crest_rules:
        thresholds["crest_factor_db"] = (
            float(crest_rules["pass"]),
            float(crest_rules["warn"]),
        )

    lra_rules = tm.get("loudness_range")
    if lra_rules:
        thresholds["lra_lu"] = (
            float(lra_rules["pass"]),
            float(lra_rules["warn"]),
        )

    dr_db_rules = tm.get("dynamic_range_db")
    if dr_db_rules:
        thresholds["dynamic_range_db"] = (
            float(dr_db_rules["pass"]),
            float(dr_db_rules["warn"]),
        )

    dr_lu_rules = tm.get("dynamic_range_lu")
    if dr_lu_rules:
        thresholds["dynamic_range_lu"] = (
            float(dr_lu_rules["pass"]),
            float(dr_lu_rules["warn"]),
        )

    tonal_rules = tm.get("tonal_peak")
    if tonal_rules:
        thresholds["tonal_peak_delta_db"] = (
            float(tonal_rules["pass"]),
            float(tonal_rules["warn"]),
        )

    clipped_samples_rules = tm.get("clipped_samples") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get(
        "clipped_samples"
    )
    if clipped_samples_rules:
        thresholds["clipped_samples"] = (
            float(clipped_samples_rules["pass"]),
            float(clipped_samples_rules["warn"]),
        )

    clipped_runs_rules = tm.get("clipped_runs") or DEFAULT_LEVEL_METRIC_THRESHOLDS.get(
        "clipped_runs"
    )
    if clipped_runs_rules:
        thresholds["clipped_runs"] = (
            float(clipped_runs_rules["pass"]),
            float(clipped_runs_rules["warn"]),
        )

    tp_rules = tm.get("true_peak_dbtp")
    if tp_rules:
        thresholds["true_peak_dbtp"] = (
            float(tp_rules["pass"]),
            float(tp_rules["warn"]),
        )
    else:
        tp_default = DEFAULT_LEVEL_METRIC_THRESHOLDS.get("true_peak_dbtp", {})
        tp = normalization.get("true_peak", {})
        if bool(tp.get("enabled", False)):
            max_dbtp = float(tp.get("max_dbtp", tp_default.get("pass", -1.0)))
            thresholds["true_peak_dbtp"] = (max_dbtp, max_dbtp + 0.5)
        elif tp_default:
            thresholds["true_peak_dbtp"] = (
                float(tp_default["pass"]),
                float(tp_default["warn"]),
            )

    if corr_rules:
        thresholds["stereo_correlation"] = {
            "frame_seconds": float(corr_rules.get("frame_seconds", 0.5)),
            "hop_seconds": float(corr_rules.get("hop_seconds", 0.25)),
            "mean": _range_cfg(corr_rules.get("mean", {}), default_min=0.0, default_max=1.0),
            "min": _range_cfg(corr_rules.get("min", {}), default_min=-1.0, default_max=1.0),
            "inversion": {
                "threshold": float(
                    corr_rules.get("inversion", {}).get("threshold", -0.8)
                ),
                "pass": float(corr_rules.get("inversion", {}).get("pass", 0.05)),
                "warn": float(corr_rules.get("inversion", {}).get("warn", 0.2)),
            },
        }

    if delay_rules:
        thresholds["inter_channel_delay"] = build_inter_channel_delay_config(delay_rules)
    if consistency_rules:
        thresholds["channel_consistency"] = build_channel_consistency_config(consistency_rules)

    noise_floor_defaults = derive_noise_floor_baselines(freqs, ref_mean, bands)
    noise_floor_baselines = {
        entry["band_name"]: float(entry["noise_floor_db"])
        for entry in j.get("noise_floor_baselines", [])
        if isinstance(entry, dict)
        and "band_name" in entry
        and "noise_floor_db" in entry
    }
    noise_floor_by_band = {**noise_floor_defaults, **noise_floor_baselines}

    algorithm_registry = j.get("algorithm_registry", {})

    return ReferenceProfile(
        name=j["profile"]["name"],
        kind=j["profile"]["kind"],
        version=j["profile"]["version"],
        profile_hash_sha256=j["integrity"]["profile_hash_sha256"],
        analysis_lock_hash=j.get("analysis_lock_hash", ""),
        algorithm_ids=list(j.get("algorithm_ids", [])),
        freqs_hz=freqs,
        ref_mean_db=ref_mean,
        bands=bands,
        thresholds=thresholds,
        analysis_lock=analysis_lock,
        algorithm_registry=algorithm_registry,
        normalization=normalization,
        noise_floor_by_band=noise_floor_by_band,
    )
