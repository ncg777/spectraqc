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
        elif tp_stat == Status.WARN:
            warn_count += 1

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
