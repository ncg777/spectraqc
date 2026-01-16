from __future__ import annotations
from spectraqc.types import (
    Status,
    ThresholdResult,
    BandDecision,
    ProgramDecision,
    BandMetrics,
    GlobalMetrics,
)


def _status_abs(value_abs: float, pass_lim: float, warn_lim: float) -> Status:
    """Evaluate status based on absolute value thresholds."""
    if value_abs <= pass_lim:
        return Status.PASS
    if value_abs <= warn_lim:
        return Status.WARN
    return Status.FAIL


def _status_high_is_bad(value: float, pass_lim: float, warn_lim: float) -> Status:
    """Evaluate status where higher values are worse (e.g., true peak)."""
    if value <= pass_lim:
        return Status.PASS
    if value <= warn_lim:
        return Status.WARN
    return Status.FAIL


def _status_low_is_bad(value: float, pass_lim: float, warn_lim: float) -> Status:
    """Evaluate status where lower values are worse (e.g., crest factor)."""
    if value >= pass_lim:
        return Status.PASS
    if value >= warn_lim:
        return Status.WARN
    return Status.FAIL


def _explain(
    *,
    metric: str,
    value: float,
    units: str,
    status: Status,
    pass_lim: float,
    warn_lim: float,
    compare: str
) -> str:
    """Build a short, human-readable explanation for WARN/FAIL."""
    if status == Status.PASS:
        return ""
    thr = f"pass<= {pass_lim:g} {units}, warn<= {warn_lim:g} {units}"
    return f"{metric} is {compare} threshold: {value:.3f} {units} ({thr})."


def evaluate(
    band_metrics: list[BandMetrics],
    global_metrics: GlobalMetrics,
    thresholds: dict
) -> ProgramDecision:
    """
    Evaluate all metrics against thresholds to produce pass/warn/fail decisions.
    
    Args:
        band_metrics: List of computed band metrics
        global_metrics: Computed global metrics
        thresholds: Threshold configuration from profile
        
    Returns:
        ProgramDecision with overall status and per-metric decisions
    """
    band_decisions: list[BandDecision] = []
    warn_count = 0
    any_fail = False

    for bm in band_metrics:
        b = bm.band
        
        # Get thresholds for this band
        pass_m, warn_m = thresholds["band_mean_db"].get(
            b.name, thresholds["band_mean_db"]["default"]
        )
        pass_x, warn_x = thresholds["band_max_db"].get(
            b.name, thresholds["band_max_db"]["all"]
        )

        mean_abs = abs(bm.mean_deviation_db)
        max_abs = bm.max_deviation_db

        # Mean deviation result
        mean_status = _status_abs(mean_abs, pass_m, warn_m)
        mean_res = ThresholdResult(
            metric="band_mean_deviation",
            value=bm.mean_deviation_db,
            units="dB",
            status=mean_status,
            pass_limit=pass_m,
            warn_limit=warn_m,
            notes=_explain(
                metric="band_mean_deviation",
                value=bm.mean_deviation_db,
                units="dB",
                status=mean_status,
                pass_lim=pass_m,
                warn_lim=warn_m,
                compare="outside absolute"
            )
        )
        
        # Max deviation result
        max_status = _status_abs(max_abs, pass_x, warn_x)
        max_res = ThresholdResult(
            metric="band_max_deviation",
            value=max_abs,
            units="dB",
            status=max_status,
            pass_limit=pass_x,
            warn_limit=warn_x,
            notes=_explain(
                metric="band_max_deviation",
                value=max_abs,
                units="dB",
                status=max_status,
                pass_lim=pass_x,
                warn_lim=warn_x,
                compare="outside absolute"
            )
        )

        # Variance ratio result
        v_pass, v_warn = thresholds["variance_ratio"]
        v = bm.variance_ratio
        v_stat = Status.PASS if v <= v_pass else (
            Status.WARN if v <= v_warn else Status.FAIL
        )
        var_res = ThresholdResult(
            metric="variance_ratio",
            value=v,
            units="ratio",
            status=v_stat,
            pass_limit=v_pass,
            warn_limit=v_warn,
            notes=_explain(
                metric="variance_ratio",
                value=v,
                units="ratio",
                status=v_stat,
                pass_lim=v_pass,
                warn_lim=v_warn,
                compare="above"
            )
        )

        bd = BandDecision(band=b, mean=mean_res, max=max_res, variance=var_res)
        band_decisions.append(bd)

        # Track warnings and failures
        for r in (mean_res, max_res, var_res):
            if r.status == Status.FAIL:
                any_fail = True
            elif r.status == Status.WARN:
                warn_count += 1

    # Global metrics evaluation
    t_pass, t_warn = thresholds["tilt_db_per_oct"]
    tilt_abs = abs(global_metrics.tilt_deviation_db_per_oct)
    tilt_stat = _status_abs(tilt_abs, t_pass, t_warn)
    
    global_decisions = [
        ThresholdResult(
            metric="tilt_deviation",
            value=global_metrics.tilt_deviation_db_per_oct,
            units="dB/oct",
            status=tilt_stat,
            pass_limit=t_pass,
            warn_limit=t_warn,
            notes=_explain(
                metric="tilt_deviation",
                value=global_metrics.tilt_deviation_db_per_oct,
                units="dB/oct",
                status=tilt_stat,
                pass_lim=t_pass,
                warn_lim=t_warn,
                compare="outside absolute"
            )
        )
    ]

    # True peak evaluation if configured
    if global_metrics.true_peak_dbtp is not None and "true_peak_dbtp" in thresholds:
        tp_pass, tp_warn = thresholds["true_peak_dbtp"]
        tp_stat = _status_high_is_bad(
            float(global_metrics.true_peak_dbtp), tp_pass, tp_warn
        )
        global_decisions.append(
            ThresholdResult(
                metric="true_peak",
                value=float(global_metrics.true_peak_dbtp),
                units="dBTP",
                status=tp_stat,
                pass_limit=tp_pass,
                warn_limit=tp_warn,
                notes=_explain(
                    metric="true_peak",
                    value=float(global_metrics.true_peak_dbtp),
                    units="dBTP",
                    status=tp_stat,
                    pass_lim=tp_pass,
                    warn_lim=tp_warn,
                    compare="above"
                )
            )
        )
        if tp_stat == Status.FAIL:
            any_fail = True

    if global_metrics.peak_dbfs is not None and "peak_dbfs" in thresholds:
        peak_pass, peak_warn = thresholds["peak_dbfs"]
        peak_stat = _status_high_is_bad(float(global_metrics.peak_dbfs), peak_pass, peak_warn)
        global_decisions.append(
            ThresholdResult(
                metric="peak_dbfs",
                value=float(global_metrics.peak_dbfs),
                units="dBFS",
                status=peak_stat,
                pass_limit=peak_pass,
                warn_limit=peak_warn,
                notes=_explain(
                    metric="peak_dbfs",
                    value=float(global_metrics.peak_dbfs),
                    units="dBFS",
                    status=peak_stat,
                    pass_lim=peak_pass,
                    warn_lim=peak_warn,
                    compare="above"
                )
            )
        )
        if peak_stat == Status.FAIL:
            any_fail = True

    if global_metrics.rms_dbfs is not None and "rms_dbfs" in thresholds:
        rms_pass, rms_warn = thresholds["rms_dbfs"]
        rms_stat = _status_high_is_bad(float(global_metrics.rms_dbfs), rms_pass, rms_warn)
        global_decisions.append(
            ThresholdResult(
                metric="rms_dbfs",
                value=float(global_metrics.rms_dbfs),
                units="dBFS",
                status=rms_stat,
                pass_limit=rms_pass,
                warn_limit=rms_warn,
                notes=_explain(
                    metric="rms_dbfs",
                    value=float(global_metrics.rms_dbfs),
                    units="dBFS",
                    status=rms_stat,
                    pass_lim=rms_pass,
                    warn_lim=rms_warn,
                    compare="above"
                )
            )
        )
        if rms_stat == Status.FAIL:
            any_fail = True

    if global_metrics.crest_factor_db is not None and "crest_factor_db" in thresholds:
        crest_pass, crest_warn = thresholds["crest_factor_db"]
        crest_stat = _status_low_is_bad(
            float(global_metrics.crest_factor_db), crest_pass, crest_warn
        )
        global_decisions.append(
            ThresholdResult(
                metric="crest_factor_db",
                value=float(global_metrics.crest_factor_db),
                units="dB",
                status=crest_stat,
                pass_limit=crest_pass,
                warn_limit=crest_warn,
                notes=_explain(
                    metric="crest_factor_db",
                    value=float(global_metrics.crest_factor_db),
                    units="dB",
                    status=crest_stat,
                    pass_lim=crest_pass,
                    warn_lim=crest_warn,
                    compare="below"
                )
            )
        )
        if crest_stat == Status.FAIL:
            any_fail = True

    if global_metrics.lra_lu is not None and "lra_lu" in thresholds:
        lra_pass, lra_warn = thresholds["lra_lu"]
        lra_stat = _status_high_is_bad(float(global_metrics.lra_lu), lra_pass, lra_warn)
        global_decisions.append(
            ThresholdResult(
                metric="loudness_range",
                value=float(global_metrics.lra_lu),
                units="LU",
                status=lra_stat,
                pass_limit=lra_pass,
                warn_limit=lra_warn,
                notes=_explain(
                    metric="loudness_range",
                    value=float(global_metrics.lra_lu),
                    units="LU",
                    status=lra_stat,
                    pass_lim=lra_pass,
                    warn_lim=lra_warn,
                    compare="above"
                )
            )
        )
        if lra_stat == Status.FAIL:
            any_fail = True

    # Tonal peak evaluation if configured
    if (
        global_metrics.tonal_peak_max_delta_db is not None
        and "tonal_peak_delta_db" in thresholds
    ):
        tone_pass, tone_warn = thresholds["tonal_peak_delta_db"]
        tone_stat = _status_high_is_bad(
            float(global_metrics.tonal_peak_max_delta_db), tone_pass, tone_warn
        )
        global_decisions.append(
            ThresholdResult(
                metric="tonal_peak",
                value=float(global_metrics.tonal_peak_max_delta_db),
                units="dB",
                status=tone_stat,
                pass_limit=tone_pass,
                warn_limit=tone_warn,
                notes=_explain(
                    metric="tonal_peak",
                    value=float(global_metrics.tonal_peak_max_delta_db),
                    units="dB",
                    status=tone_stat,
                    pass_lim=tone_pass,
                    warn_lim=tone_warn,
                    compare="above"
                )
            )
        )
        if tone_stat == Status.FAIL:
            any_fail = True

    # Overall status determination
    warn_threshold = thresholds.get("warn_if_warn_band_count_at_least", 2)
    overall = (
        Status.FAIL if any_fail else (
            Status.WARN if warn_count >= warn_threshold else Status.PASS
        )
    )
    
    return ProgramDecision(
        overall_status=overall,
        band_decisions=band_decisions,
        global_decisions=global_decisions
    )
