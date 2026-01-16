from __future__ import annotations


DEFAULT_SPECTRAL_ARTIFACT_THRESHOLDS = {
    "cutoff": {
        "min_hz": 4000.0,
        "drop_db": 24.0,
        "window_bins": 6,
        "hold_bins": 8,
        "warn_fraction": 0.9,
        "fail_fraction": 0.8,
    },
    "mirror": {
        "min_bins": 8,
        "warn_similarity": 0.75,
        "fail_similarity": 0.9,
        "min_flatness": 0.15,
    },
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


def build_spectral_artifact_config(overrides: dict | None = None) -> dict:
    """Return merged spectral artifact configuration with defaults applied."""
    return _merge_config(DEFAULT_SPECTRAL_ARTIFACT_THRESHOLDS, overrides)


def evaluate_spectral_artifacts(
    metrics: dict,
    *,
    expected_max_hz: float,
    config: dict | None = None
) -> list[dict]:
    """
    Evaluate spectral artifacts against thresholds to produce rule flags.

    Returns:
        List of flag dicts with rule identifiers and measurements.
    """
    cfg = build_spectral_artifact_config(config)
    flags: list[dict] = []

    cutoff_freq = metrics.get("cutoff_freq_hz")
    cutoff_drop_db = metrics.get("cutoff_drop_db")
    if cutoff_freq is not None and expected_max_hz:
        cutoff_ratio = float(cutoff_freq) / float(expected_max_hz)
        warn_frac = float(cfg["cutoff"]["warn_fraction"])
        fail_frac = float(cfg["cutoff"]["fail_fraction"])
        status = None
        if cutoff_ratio < fail_frac:
            status = "fail"
        elif cutoff_ratio < warn_frac:
            status = "warn"
        if status:
            flags.append(
                {
                    "rule_id": "brickwall_cutoff_below_profile",
                    "status": status,
                    "measurements": {
                        "cutoff_freq_hz": float(cutoff_freq),
                        "cutoff_ratio": cutoff_ratio,
                        "cutoff_drop_db": float(cutoff_drop_db) if cutoff_drop_db is not None else None,
                        "expected_max_hz": float(expected_max_hz),
                    },
                }
            )

    mirror_similarity = metrics.get("mirror_similarity")
    mirror_flatness = metrics.get("mirror_flatness")
    if mirror_similarity is not None:
        warn_similarity = float(cfg["mirror"]["warn_similarity"])
        fail_similarity = float(cfg["mirror"]["fail_similarity"])
        min_flatness = cfg["mirror"].get("min_flatness")
        flatness_ok = True
        if min_flatness is not None and mirror_flatness is not None:
            flatness_ok = float(mirror_flatness) >= float(min_flatness)
        if flatness_ok:
            status = None
            if mirror_similarity >= fail_similarity:
                status = "fail"
            elif mirror_similarity >= warn_similarity:
                status = "warn"
            if status:
                flags.append(
                    {
                        "rule_id": "mirrored_spectral_images",
                        "status": status,
                        "measurements": {
                            "mirror_similarity": float(mirror_similarity),
                            "mirror_flatness": float(mirror_flatness) if mirror_flatness is not None else None,
                            "mirror_pivot_hz": metrics.get("mirror_pivot_hz"),
                            "mirror_band_hz": metrics.get("mirror_band_hz"),
                        },
                    }
                )

    return flags
