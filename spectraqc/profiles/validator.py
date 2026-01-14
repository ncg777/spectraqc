"""Reference profile validation helpers."""
from __future__ import annotations
from typing import Any
import math


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)


def validate_reference_profile_dict(j: dict) -> None:
    """Validate reference profile structure and core constraints."""
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    for k in ("profile", "frequency_grid", "reference_curves", "bands", "analysis_lock", "threshold_model", "integrity"):
        if k not in j:
            err(f"missing key: {k}")

    if errors:
        raise ValueError("; ".join(errors))

    freqs = j["frequency_grid"].get("freqs_hz")
    if not isinstance(freqs, list) or len(freqs) < 2:
        err("frequency_grid.freqs_hz must be a list with at least 2 entries.")
        freqs = []
    else:
        last = None
        for i, f in enumerate(freqs):
            if not _is_number(f):
                err(f"frequency_grid.freqs_hz[{i}] must be a finite number.")
                break
            if last is not None and f <= last:
                err("frequency_grid.freqs_hz must be strictly increasing.")
                break
            last = f

    ref_curves = j.get("reference_curves", {})
    mean_db = ref_curves.get("mean_db")
    if not isinstance(mean_db, list) or len(mean_db) != len(freqs):
        err("reference_curves.mean_db must match frequency grid length.")
    else:
        for i, v in enumerate(mean_db):
            if not _is_number(v):
                err(f"reference_curves.mean_db[{i}] must be a finite number.")
                break

    var_db2 = ref_curves.get("var_db2")
    if var_db2 is not None:
        if not isinstance(var_db2, list) or len(var_db2) != len(freqs):
            err("reference_curves.var_db2 must match frequency grid length.")
        else:
            for i, v in enumerate(var_db2):
                if not _is_number(v) or v < 0:
                    err(f"reference_curves.var_db2[{i}] must be a non-negative number.")
                    break

    bands = j.get("bands", [])
    if not isinstance(bands, list) or not bands:
        err("bands must be a non-empty list.")
    else:
        band_names: set[str] = set()
        f_min = freqs[0] if freqs else 0.0
        f_max = freqs[-1] if freqs else 0.0
        for i, b in enumerate(bands):
            if not isinstance(b, dict):
                err(f"bands[{i}] must be an object.")
                continue
            name = b.get("name")
            f_low = b.get("f_low_hz")
            f_high = b.get("f_high_hz")
            if not isinstance(name, str) or not name:
                err(f"bands[{i}].name must be a non-empty string.")
            elif name in band_names:
                err(f"bands[{i}].name is duplicated: {name}")
            else:
                band_names.add(name)
            if not _is_number(f_low) or not _is_number(f_high) or f_high <= f_low:
                err(f"bands[{i}] must have f_low_hz < f_high_hz.")
            if _is_number(f_low) and f_low < f_min:
                err(f"bands[{i}].f_low_hz below frequency grid minimum.")
            if _is_number(f_high) and f_high > f_max:
                err(f"bands[{i}].f_high_hz above frequency grid maximum.")

    rules = j.get("threshold_model", {}).get("rules", {})
    band_mean = rules.get("band_mean", {}).get("default", {})
    band_max = rules.get("band_max", {}).get("default", {})
    tilt = rules.get("tilt", {})
    for name, obj in (("band_mean", band_mean), ("band_max", band_max), ("tilt", tilt)):
        p = obj.get("pass")
        w = obj.get("warn")
        if not _is_number(p) or not _is_number(w) or w < p:
            err(f"threshold_model.rules.{name} pass/warn must be numbers with warn>=pass.")
    by_band = rules.get("band_mean", {}).get("by_band", [])
    if isinstance(by_band, list):
        valid_names = {b.get("name") for b in bands if isinstance(b, dict)}
        for i, bb in enumerate(by_band):
            bname = bb.get("band_name")
            if bname not in valid_names:
                err(f"threshold_model.rules.band_mean.by_band[{i}] has unknown band_name.")

    agg = j.get("threshold_model", {}).get("aggregation", {})
    warn_count = agg.get("warn_if_warn_band_count_at_least")
    if warn_count is None or not isinstance(warn_count, int) or warn_count < 0:
        err("threshold_model.aggregation.warn_if_warn_band_count_at_least must be a non-negative int.")

    analysis_lock = j.get("analysis_lock", {})
    channel_policy = analysis_lock.get("channel_policy", "mono")
    if channel_policy not in ("mono", "stereo_average", "mid_only", "per_channel"):
        err("analysis_lock.channel_policy must be one of mono, stereo_average, mid_only, per_channel.")

    smoothing = analysis_lock.get("smoothing", {"type": "none"})
    if smoothing.get("type") not in ("none", "octave_fraction", "log_hz"):
        err("analysis_lock.smoothing.type must be none, octave_fraction, or log_hz.")
    if smoothing.get("type") == "octave_fraction":
        oct_frac = smoothing.get("octave_fraction")
        if not _is_number(oct_frac) or oct_frac <= 0:
            err("analysis_lock.smoothing.octave_fraction must be > 0.")
    if smoothing.get("type") == "log_hz":
        bins = smoothing.get("log_hz_bins_per_octave")
        if not isinstance(bins, int) or bins <= 0:
            err("analysis_lock.smoothing.log_hz_bins_per_octave must be a positive int.")

    normalization = analysis_lock.get("normalization", {})
    loud = normalization.get("loudness", {})
    tp = normalization.get("true_peak", {})
    if not isinstance(loud.get("enabled", False), bool):
        err("analysis_lock.normalization.loudness.enabled must be boolean.")
    if loud.get("enabled", False):
        if not _is_number(loud.get("target_lufs_i")):
            err("analysis_lock.normalization.loudness.target_lufs_i must be number.")
        if not isinstance(loud.get("algorithm_id", ""), str) or not loud.get("algorithm_id"):
            err("analysis_lock.normalization.loudness.algorithm_id required when enabled.")
    if not isinstance(tp.get("enabled", False), bool):
        err("analysis_lock.normalization.true_peak.enabled must be boolean.")
    if tp.get("enabled", False):
        if not _is_number(tp.get("max_dbtp")):
            err("analysis_lock.normalization.true_peak.max_dbtp must be number.")
        if not isinstance(tp.get("algorithm_id", ""), str) or not tp.get("algorithm_id"):
            err("analysis_lock.normalization.true_peak.algorithm_id required when enabled.")

    if errors:
        raise ValueError("; ".join(errors))
