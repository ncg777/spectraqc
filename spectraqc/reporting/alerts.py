from __future__ import annotations


_METRIC_LABELS = {
    "band_mean_deviation": "Band mean deviation",
    "band_max_deviation": "Band max deviation",
    "variance_ratio": "Variance ratio",
    "tilt_deviation": "Spectral tilt deviation",
    "true_peak": "True peak",
    "peak_dbfs": "Peak level",
    "rms_dbfs": "RMS level",
    "lufs_i": "Integrated loudness",
    "noise_floor": "Noise floor",
    "crest_factor_db": "Crest factor",
    "loudness_range": "Loudness range",
    "dynamic_range_db": "Dynamic range (dB)",
    "dynamic_range_lu": "Dynamic range (LU)",
    "tonal_peak": "Tonal peak delta",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " "))


def _format_alert_message(decision: dict, *, band_name: str | None = None) -> str:
    notes = decision.get("notes")
    if notes:
        if band_name:
            return f"{band_name} band: {notes}"
        return notes
    metric = _metric_label(str(decision.get("metric", "metric")))
    status = decision.get("status", "warn")
    value = decision.get("value")
    units = decision.get("units", "")
    pass_limit = decision.get("pass_limit")
    warn_limit = decision.get("warn_limit")
    details = ""
    if isinstance(value, (int, float)):
        details = f": {value:.3f} {units}".rstrip()
    if isinstance(pass_limit, (int, float)) and isinstance(warn_limit, (int, float)):
        details += f" (pass<= {pass_limit:g} {units}, warn<= {warn_limit:g} {units})"
    prefix = f"{metric}{details}."
    if band_name:
        prefix = f"{band_name} band {metric}{details}."
    return f"{prefix} Status: {status}."


def build_alerts(decisions: dict) -> list[dict]:
    """Build alert list from decision results."""
    alerts: list[dict] = []
    for bd in decisions.get("band_decisions", []):
        band_name = bd.get("band_name")
        for key in ("mean", "max", "variance"):
            decision = bd.get(key, {})
            status = decision.get("status")
            if status in ("warn", "fail"):
                metric = decision.get("metric", key)
                alerts.append(
                    {
                        "scope": "band",
                        "band_name": band_name,
                        "metric": metric,
                        "status": status,
                        "message": _format_alert_message(
                            decision, band_name=str(band_name) if band_name else None
                        ),
                    }
                )
    for gd in decisions.get("global_decisions", []):
        status = gd.get("status")
        if status in ("warn", "fail"):
            metric = gd.get("metric")
            alerts.append(
                {
                    "scope": "global",
                    "metric": metric,
                    "status": status,
                    "message": _format_alert_message(gd),
                }
            )
    return alerts
