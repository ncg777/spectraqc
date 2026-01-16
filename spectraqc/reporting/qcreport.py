from __future__ import annotations
import numpy as np
from spectraqc.reporting.alerts import build_alerts
from spectraqc.utils.hashing import sha256_hex_canonical_json
from spectraqc.utils.quantize import q, q_list


def _tolist(a: np.ndarray) -> list[float]:
    """Convert numpy array to list of floats."""
    return [float(x) for x in a.tolist()]


def build_qcreport_dict(
    *,
    engine: dict,
    input_meta: dict,
    profile: dict,
    analysis: dict,
    freqs_hz: np.ndarray,
    ref_mean_db: np.ndarray,
    ref_var_db2: np.ndarray,
    ltpsd_mean_db: np.ndarray,
    ltpsd_var_db2: np.ndarray,
    delta_mean_db: np.ndarray,
    band_metrics: list[dict],
    global_metrics: dict,
    noise_floor: dict | None = None,
    decisions: dict,
    confidence: dict,
    repair: dict | None = None,
    cohort_id: str | None = None,
    department: str | None = None,
    campaign: str | None = None
) -> dict:
    """
    Build a QCReport dictionary with proper quantization and integrity hash.
    
    Args:
        engine: Engine metadata (name, version, build info)
        input_meta: Input file metadata
        profile: Profile metadata
        analysis: Analysis configuration
        freqs_hz: Frequency grid
        ltpsd_mean_db: Long-term PSD mean in dB
        ltpsd_var_db2: Long-term PSD variance in dB²
        delta_mean_db: Deviation curve in dB
        band_metrics: List of band metrics dictionaries
        global_metrics: Global metrics dictionary
        decisions: Decision results dictionary
        confidence: Confidence assessment dictionary
        
    Returns:
        Complete QCReport dictionary with integrity hash
    """
    report = {
        "schema_version": "1.0",
        "report_id": analysis.get("report_id", "qc_local"),
        "created_utc": analysis.get("created_utc", "1970-01-01T00:00:00Z"),
        "engine": engine,
        "input": input_meta,
        "profile": profile,
        "analysis": analysis,
        "metrics": {
            "frequency_grid": {
                "grid_kind": "fft_one_sided",
                "units": "Hz",
                "freqs_hz": _tolist(freqs_hz)
            },
            "ltpsd": {
                "mean_db": _tolist(ltpsd_mean_db),
                "var_db2": _tolist(ltpsd_var_db2)
            },
            "reference": {
                "mean_db": _tolist(ref_mean_db),
                "var_db2": _tolist(ref_var_db2),
            },
            "deviation": {
                "delta_mean_db": _tolist(delta_mean_db)
            },
            "band_metrics": band_metrics,
            "global_metrics": global_metrics
        },
        "decisions": decisions,
        "confidence": confidence,
        "integrity": {
            "qcreport_hash_sha256": "",
            "signed": False,
            "signature": {"algo": "none", "value_b64": ""}
        }
    }

    alerts = build_alerts(decisions)
    if alerts:
        report["alerts"] = alerts

    if repair is not None:
        report["repair"] = repair

    if noise_floor is not None:
        report["metrics"]["noise_floor"] = noise_floor

    cohort_meta = {
        "cohort_id": cohort_id,
        "department": department,
        "campaign": campaign
    }
    cohort_meta = {k: v for k, v in cohort_meta.items() if v}
    if cohort_meta:
        report["cohort"] = cohort_meta

    # Quantize for stable hashing
    if "normalization" in report.get("analysis", {}):
        l = report["analysis"]["normalization"].get("loudness", {})
        if "measured_lufs_i" in l:
            l["measured_lufs_i"] = q(float(l["measured_lufs_i"]), 0.01)
        if "applied_gain_db" in l:
            l["applied_gain_db"] = q(float(l["applied_gain_db"]), 0.01)

        tp = report["analysis"]["normalization"].get("true_peak", {})
        if "measured_dbtp" in tp:
            tp["measured_dbtp"] = q(float(tp["measured_dbtp"]), 0.01)

    report["metrics"]["ltpsd"]["mean_db"] = q_list(
        report["metrics"]["ltpsd"]["mean_db"], 0.01
    )
    report["metrics"]["deviation"]["delta_mean_db"] = q_list(
        report["metrics"]["deviation"]["delta_mean_db"], 0.01
    )
    report["metrics"]["ltpsd"]["var_db2"] = q_list(
        report["metrics"]["ltpsd"]["var_db2"], 0.001
    )
    report["metrics"]["reference"]["mean_db"] = q_list(
        report["metrics"]["reference"]["mean_db"], 0.01
    )
    report["metrics"]["reference"]["var_db2"] = q_list(
        report["metrics"]["reference"]["var_db2"], 0.001
    )

    for bm in report["metrics"]["band_metrics"]:
        bm["mean_deviation_db"] = q(float(bm["mean_deviation_db"]), 0.01)
        bm["max_deviation_db"] = q(float(bm["max_deviation_db"]), 0.01)
        bm["variance_ratio"] = q(float(bm["variance_ratio"]), 0.001)

    gm = report["metrics"]["global_metrics"]
    if "spectral_tilt_db_per_oct" in gm:
        gm["spectral_tilt_db_per_oct"] = q(float(gm["spectral_tilt_db_per_oct"]), 0.001)
    if "tilt_deviation_db_per_oct" in gm:
        gm["tilt_deviation_db_per_oct"] = q(float(gm["tilt_deviation_db_per_oct"]), 0.001)
    if "true_peak_dbtp" in gm:
        gm["true_peak_dbtp"] = q(float(gm["true_peak_dbtp"]), 0.01)
    if "peak_dbfs" in gm:
        gm["peak_dbfs"] = q(float(gm["peak_dbfs"]), 0.01)
    if "rms_dbfs" in gm:
        gm["rms_dbfs"] = q(float(gm["rms_dbfs"]), 0.01)
    if "lufs_i" in gm:
        gm["lufs_i"] = q(float(gm["lufs_i"]), 0.01)
    if "crest_factor_db" in gm:
        gm["crest_factor_db"] = q(float(gm["crest_factor_db"]), 0.01)
    if "lra_lu" in gm:
        gm["lra_lu"] = q(float(gm["lra_lu"]), 0.01)
    if "dynamic_range_db" in gm:
        gm["dynamic_range_db"] = q(float(gm["dynamic_range_db"]), 0.01)
    if "dynamic_range_lu" in gm:
        gm["dynamic_range_lu"] = q(float(gm["dynamic_range_lu"]), 0.01)
    if "tonal_peak_max_delta_db" in gm:
        gm["tonal_peak_max_delta_db"] = q(float(gm["tonal_peak_max_delta_db"]), 0.01)
    if "noise_floor_dbfs" in gm:
        gm["noise_floor_dbfs"] = q(float(gm["noise_floor_dbfs"]), 0.01)
    if "tonal_peaks" in gm:
        for peak in gm["tonal_peaks"]:
            if "frequency_hz" in peak:
                peak["frequency_hz"] = q(float(peak["frequency_hz"]), 0.1)
            if "level_db" in peak:
                peak["level_db"] = q(float(peak["level_db"]), 0.01)
            if "noise_floor_db" in peak and peak["noise_floor_db"] is not None:
                peak["noise_floor_db"] = q(float(peak["noise_floor_db"]), 0.01)
            if "delta_db" in peak and peak["delta_db"] is not None:
                peak["delta_db"] = q(float(peak["delta_db"]), 0.01)
    if "level_anomalies" in gm:
        def _quantize_segment_block(block: dict | None) -> None:
            if not block:
                return
            if "total_duration_s" in block:
                block["total_duration_s"] = q(float(block["total_duration_s"]), 0.001)
            for seg in block.get("segments", []):
                if "start_s" in seg:
                    seg["start_s"] = q(float(seg["start_s"]), 0.001)
                if "end_s" in seg:
                    seg["end_s"] = q(float(seg["end_s"]), 0.001)
                if "duration_s" in seg:
                    seg["duration_s"] = q(float(seg["duration_s"]), 0.001)
                if "baseline_dbfs" in seg:
                    seg["baseline_dbfs"] = q(float(seg["baseline_dbfs"]), 0.01)
                if "level_dbfs" in seg:
                    seg["level_dbfs"] = q(float(seg["level_dbfs"]), 0.01)
                if "drop_db" in seg:
                    seg["drop_db"] = q(float(seg["drop_db"]), 0.01)

        level = gm["level_anomalies"]
        _quantize_segment_block(level.get("drop"))
        _quantize_segment_block(level.get("zero"))
        for channel in level.get("channels", []):
            _quantize_segment_block(channel.get("drop"))
            _quantize_segment_block(channel.get("zero"))

    if report.get("repair"):
        repair_metrics = report["repair"].get("metrics", {})
        for key in ("before", "after", "delta"):
            section = repair_metrics.get(key, {})
            if "true_peak_dbtp" in section:
                section["true_peak_dbtp"] = q(float(section["true_peak_dbtp"]), 0.01)
            if "noise_floor_dbfs" in section:
                section["noise_floor_dbfs"] = q(float(section["noise_floor_dbfs"]), 0.01)
            if "deviation_curve_db" in section:
                section["deviation_curve_db"] = q_list(section["deviation_curve_db"], 0.01)

    noise_floor = report["metrics"].get("noise_floor")
    if noise_floor:
        if "merged_dbfs" in noise_floor and noise_floor["merged_dbfs"] is not None:
            noise_floor["merged_dbfs"] = q(float(noise_floor["merged_dbfs"]), 0.01)
        if "by_channel_dbfs" in noise_floor and noise_floor["by_channel_dbfs"] is not None:
            noise_floor["by_channel_dbfs"] = q_list(noise_floor["by_channel_dbfs"], 0.01)

    # Compute integrity hash (excluding integrity object itself)
    tmp = dict(report)
    tmp.pop("integrity", None)
    report["integrity"]["qcreport_hash_sha256"] = sha256_hex_canonical_json(tmp)
    
    return report
