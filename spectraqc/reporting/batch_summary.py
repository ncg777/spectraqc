from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import io
from typing import Iterable

import numpy as np

from spectraqc.utils.hashing import sha256_hex_canonical_json


def _safe_date(created_utc: str | None) -> str:
    if not created_utc:
        return "unknown"
    return str(created_utc)[:10]


def _summary_stats(values: Iterable[float]) -> dict | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _band_status(band_decision: dict) -> str:
    statuses = []
    for key in ("mean", "max", "variance"):
        status = band_decision.get(key, {}).get("status")
        if status:
            statuses.append(status)
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _report_band_average(report: dict, field: str) -> float | None:
    metrics = report.get("metrics", {}).get("band_metrics", [])
    vals = [m.get(field) for m in metrics if isinstance(m.get(field), (int, float))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def build_batch_summary(
    results: list[tuple[str, str, str | None, dict | None]],
    *,
    generated_utc: str | None = None
) -> dict:
    """Build a cohort summary across QC reports."""
    counts = {"pass": 0, "warn": 0, "fail": 0, "error": 0}
    confidence_counts = {"pass": 0, "warn": 0, "fail": 0}
    confidence_reasons: dict[str, int] = {}
    failure_causes: dict[str, int] = {}
    band_counts: dict[str, dict[str, int]] = {}
    metric_values: dict[str, list[float]] = defaultdict(list)
    trend_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    confidence_trends: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0})
    cohort_values = {"cohort_id": set(), "department": set(), "campaign": set()}

    audio_hashes: list[str] = []
    profile_hashes: list[str] = []
    analysis_lock_hashes: list[str] = []

    for _, status, err, report in results:
        if status not in counts:
            counts["error"] += 1
        else:
            counts[status] += 1
        if err or not report:
            if err:
                failure_causes[err] = failure_causes.get(err, 0) + 1
            continue

        created_utc = report.get("created_utc") or report.get("analysis", {}).get("created_utc")
        date_key = _safe_date(created_utc)

        input_meta = report.get("input", {})
        file_hash = input_meta.get("file_hash_sha256")
        if isinstance(file_hash, str):
            audio_hashes.append(file_hash)

        profile_meta = report.get("profile", {})
        profile_hash = profile_meta.get("profile_hash_sha256")
        if isinstance(profile_hash, str):
            profile_hashes.append(profile_hash)
        analysis_lock_hash = profile_meta.get("analysis_lock_hash")
        if isinstance(analysis_lock_hash, str):
            analysis_lock_hashes.append(analysis_lock_hash)

        cohort_meta = report.get("cohort", {})
        for key in cohort_values:
            value = cohort_meta.get(key)
            if value:
                cohort_values[key].add(str(value))

        conf = report.get("confidence", {})
        conf_status = conf.get("status", "pass")
        if conf_status in confidence_counts:
            confidence_counts[conf_status] += 1
            confidence_trends[date_key][conf_status] += 1
        for reason in conf.get("reasons", []):
            confidence_reasons[reason] = confidence_reasons.get(reason, 0) + 1

        decisions = report.get("decisions", {})
        for bd in decisions.get("band_decisions", []):
            band_name = bd.get("band_name", "unknown")
            band_status = _band_status(bd)
            band_counts.setdefault(band_name, {"pass": 0, "warn": 0, "fail": 0, "total": 0})
            band_counts[band_name][band_status] += 1
            band_counts[band_name]["total"] += 1
            if band_status in ("warn", "fail"):
                for key in ("mean", "max", "variance"):
                    d = bd.get(key, {})
                    note = d.get("notes") or d.get("metric", "unknown")
                    failure_causes[note] = failure_causes.get(note, 0) + 1

        for gd in decisions.get("global_decisions", []):
            gd_status = gd.get("status")
            if gd_status in ("warn", "fail"):
                note = gd.get("notes") or gd.get("metric", "unknown")
                failure_causes[note] = failure_causes.get(note, 0) + 1

        global_metrics = report.get("metrics", {}).get("global_metrics", {})
        tilt_dev = global_metrics.get("tilt_deviation_db_per_oct")
        if isinstance(tilt_dev, (int, float)):
            metric_values["tilt_deviation_db_per_oct"].append(float(tilt_dev))
            trend_metrics[date_key]["tilt_deviation_db_per_oct"].append(float(tilt_dev))
        true_peak = global_metrics.get("true_peak_dbtp")
        if isinstance(true_peak, (int, float)):
            metric_values["true_peak_dbtp"].append(float(true_peak))
            trend_metrics[date_key]["true_peak_dbtp"].append(float(true_peak))

        for metric_key in ("mean_deviation_db", "max_deviation_db", "variance_ratio"):
            avg = _report_band_average(report, metric_key)
            if avg is not None:
                metric_values[f"band_{metric_key}"].append(avg)
                trend_metrics[date_key][f"band_{metric_key}"].append(avg)

    band_failure_rates = []
    for band_name in sorted(band_counts.keys()):
        stats = band_counts[band_name]
        total = max(1, stats["total"])
        warn_fail = stats["warn"] + stats["fail"]
        band_failure_rates.append({
            "band_name": band_name,
            "pass": stats["pass"],
            "warn": stats["warn"],
            "fail": stats["fail"],
            "total": stats["total"],
            "warn_or_fail_rate": float(warn_fail) / float(total)
        })

    trend_series: dict[str, list[dict]] = {}
    for metric, by_date in trend_metrics.items():
        rows = []
        for date_key in sorted(by_date.keys()):
            stats = _summary_stats(by_date[date_key])
            if not stats:
                continue
            rows.append({
                "date": date_key,
                "count": stats["count"],
                "mean": stats["mean"],
                "p50": stats["p50"],
            })
        trend_series[metric] = rows

    confidence_trend_rows = []
    for date_key in sorted(confidence_trends.keys()):
        row = confidence_trends[date_key]
        confidence_trend_rows.append({
            "date": date_key,
            "pass": row.get("pass", 0),
            "warn": row.get("warn", 0),
            "fail": row.get("fail", 0),
        })

    total_reports = sum(counts.values())
    processed_reports = counts["pass"] + counts["warn"] + counts["fail"]
    denom = max(1, total_reports)
    denom_conf = max(1, sum(confidence_counts.values()))
    kpis = {
        "total_reports": total_reports,
        "processed_reports": processed_reports,
        "pass_rate": counts["pass"] / denom,
        "warn_rate": counts["warn"] / denom,
        "fail_rate": counts["fail"] / denom,
        "error_rate": counts["error"] / denom,
        "confidence_pass_rate": confidence_counts["pass"] / denom_conf,
        "confidence_warn_rate": confidence_counts["warn"] / denom_conf,
        "confidence_fail_rate": confidence_counts["fail"] / denom_conf,
    }
    for metric, values in metric_values.items():
        stats = _summary_stats(values)
        if stats:
            kpis[f"mean_{metric}"] = stats["mean"]

    audio_hashes_sorted = sorted(set(h for h in audio_hashes if h))
    profile_hashes_sorted = sorted(set(h for h in profile_hashes if h))
    analysis_lock_hashes_sorted = sorted(set(h for h in analysis_lock_hashes if h))
    checksum_payload = {
        "audio_hashes": audio_hashes_sorted,
        "profile_hashes": profile_hashes_sorted,
        "analysis_lock_hashes": analysis_lock_hashes_sorted,
    }
    checksum = {
        "audio_hashes_sha256": sha256_hex_canonical_json(audio_hashes_sorted),
        "profile_hashes_sha256": sha256_hex_canonical_json(profile_hashes_sorted),
        "analysis_lock_hashes_sha256": sha256_hex_canonical_json(analysis_lock_hashes_sorted),
        "corpus_checksum_sha256": sha256_hex_canonical_json(checksum_payload),
        "audio_files": len(audio_hashes_sorted),
        "profiles": len(profile_hashes_sorted),
    }

    cohort_meta = {}
    for key, values in cohort_values.items():
        if not values:
            continue
        if len(values) == 1:
            cohort_meta[key] = next(iter(values))
        else:
            cohort_meta[key] = "mixed"
    if cohort_meta:
        cohort_meta["values"] = {k: sorted(v) for k, v in cohort_values.items() if v}

    generated_utc = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "schema_version": "1.0",
        "generated_utc": generated_utc,
        "cohort": cohort_meta,
        "totals": {
            "reports": total_reports,
            "processed": processed_reports,
            "status_counts": counts
        },
        "confidence": {
            "counts": confidence_counts,
            "trend_by_date": confidence_trend_rows,
            "top_reasons": dict(sorted(confidence_reasons.items(), key=lambda kv: kv[1], reverse=True))
        },
        "failure_causes": dict(sorted(failure_causes.items(), key=lambda kv: kv[1], reverse=True)),
        "band_failure_rates": band_failure_rates,
        "drift_over_time": trend_series,
        "kpis": kpis,
        "checksum": checksum
    }


def build_kpi_payload(summary: dict) -> dict:
    """Flatten KPI export payload for dashboards and audits."""
    return {
        "generated_utc": summary.get("generated_utc"),
        "kpis": summary.get("kpis", {}),
        "checksum": summary.get("checksum", {}),
        "cohort": summary.get("cohort", {})
    }


def render_kpis_csv(summary: dict) -> str:
    """Render KPIs as CSV for dashboard ingestion."""
    payload = build_kpi_payload(summary)
    kpis = payload.get("kpis", {})
    checksum = payload.get("checksum", {})
    cohort = payload.get("cohort", {})

    fieldnames = [
        "generated_utc",
        "cohort_id",
        "department",
        "campaign",
        "total_reports",
        "processed_reports",
        "pass_rate",
        "warn_rate",
        "fail_rate",
        "error_rate",
        "confidence_pass_rate",
        "confidence_warn_rate",
        "confidence_fail_rate",
        "mean_tilt_deviation_db_per_oct",
        "mean_true_peak_dbtp",
        "mean_band_mean_deviation_db",
        "mean_band_max_deviation_db",
        "mean_band_variance_ratio",
        "audio_hashes_sha256",
        "profile_hashes_sha256",
        "analysis_lock_hashes_sha256",
        "corpus_checksum_sha256",
    ]

    row = {
        "generated_utc": payload.get("generated_utc"),
        "cohort_id": cohort.get("cohort_id"),
        "department": cohort.get("department"),
        "campaign": cohort.get("campaign"),
        "total_reports": kpis.get("total_reports"),
        "processed_reports": kpis.get("processed_reports"),
        "pass_rate": kpis.get("pass_rate"),
        "warn_rate": kpis.get("warn_rate"),
        "fail_rate": kpis.get("fail_rate"),
        "error_rate": kpis.get("error_rate"),
        "confidence_pass_rate": kpis.get("confidence_pass_rate"),
        "confidence_warn_rate": kpis.get("confidence_warn_rate"),
        "confidence_fail_rate": kpis.get("confidence_fail_rate"),
        "mean_tilt_deviation_db_per_oct": kpis.get("mean_tilt_deviation_db_per_oct"),
        "mean_true_peak_dbtp": kpis.get("mean_true_peak_dbtp"),
        "mean_band_mean_deviation_db": kpis.get("mean_band_mean_deviation_db"),
        "mean_band_max_deviation_db": kpis.get("mean_band_max_deviation_db"),
        "mean_band_variance_ratio": kpis.get("mean_band_variance_ratio"),
        "audio_hashes_sha256": checksum.get("audio_hashes_sha256"),
        "profile_hashes_sha256": checksum.get("profile_hashes_sha256"),
        "analysis_lock_hashes_sha256": checksum.get("analysis_lock_hashes_sha256"),
        "corpus_checksum_sha256": checksum.get("corpus_checksum_sha256"),
    }

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()
