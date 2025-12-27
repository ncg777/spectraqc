from __future__ import annotations
import json
import numpy as np
from spectraqc.types import ReferenceProfile, FrequencyBand


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
    }

    # True peak threshold if enabled
    tp = normalization.get("true_peak", {})
    if bool(tp.get("enabled", False)):
        max_dbtp = float(tp.get("max_dbtp", -1.0))
        thresholds["true_peak_dbtp"] = (max_dbtp, max_dbtp + 0.5)

    return ReferenceProfile(
        name=j["profile"]["name"],
        kind=j["profile"]["kind"],
        version=j["profile"]["version"],
        profile_hash_sha256=j["integrity"]["profile_hash_sha256"],
        analysis_lock_hash=j.get("analysis_lock_hash", ""),
        algorithm_ids=[],
        freqs_hz=freqs,
        ref_mean_db=ref_mean,
        bands=bands,
        thresholds=thresholds,
        analysis_lock=analysis_lock,
        normalization=normalization,
    )
