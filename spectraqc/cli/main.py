"""SpectraQC CLI - Spectral Quality Control Tool."""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import sys
import platform
import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from spectraqc.version import __version__
from spectraqc.types import Status, GlobalMetrics
from spectraqc.io.audio import load_audio, apply_channel_policy
from spectraqc.analysis.ltpsd import compute_ltpsd
from spectraqc.metrics.grid import interp_to_grid, interp_var_ratio
from spectraqc.metrics.smoothing import smooth_octave_fraction, smooth_log_hz
from spectraqc.metrics.deviation import deviation_curve_db
from spectraqc.metrics.integration import band_metrics
from spectraqc.metrics.tilt import spectral_tilt_db_per_oct
from spectraqc.metrics.brickwall import detect_spectral_artifacts
from spectraqc.metrics.tonal import detect_tonal_peaks
from spectraqc.metrics.truepeak import true_peak_dbtp_mono
from spectraqc.metrics.noise import noise_floor_dbfs_mono
from spectraqc.metrics.level_anomalies import detect_level_anomalies
from spectraqc.metrics.transient_spikes import detect_transient_spikes
from spectraqc.metrics.levels import (
    peak_dbfs_mono,
    rms_dbfs_mono,
    crest_factor_db_mono,
)
from spectraqc.metrics.silence import detect_silence_segments
from spectraqc.metrics.dr import (
    dynamic_range_percentile_dbfs_mono,
    dynamic_range_short_term_lufs_mono,
)
from spectraqc.metrics.loudness import loudness_metrics_mono
from spectraqc.dsp.repair import apply_repair_plan, compute_repair_metrics
from spectraqc.algorithms.registry import (
    build_algorithm_registry,
    algorithm_ids_from_registry,
    LOUDNESS_ALGO_ID,
    TRUE_PEAK_ALGO_ID,
)
from spectraqc.profiles.loader import load_reference_profile
from spectraqc.thresholds.evaluator import evaluate
from spectraqc.thresholds.brickwall import evaluate_spectral_artifacts
from spectraqc.thresholds.level_anomalies import evaluate_level_anomalies
from spectraqc.thresholds.silence_detection import (
    evaluate_silence_gaps,
    summarize_silence_gaps,
)
from spectraqc.thresholds.transient_spikes import build_transient_spike_config
from spectraqc.reporting.qcreport import build_qcreport_dict
from spectraqc.reporting.batch_summary import (
    build_batch_summary,
    build_kpi_payload,
    render_kpis_csv,
)
from spectraqc.utils.canonical_json import canonical_dumps
from spectraqc.utils.hashing import sha256_hex_file


# Exit codes per spec
EXIT_PASS = 0
EXIT_WARN = 10
EXIT_FAIL = 20
EXIT_BAD_ARGS = 2
EXIT_DECODE_ERROR = 3
EXIT_PROFILE_ERROR = 4
EXIT_INTERNAL_ERROR = 5
MIN_EFFECTIVE_SECONDS = 0.5
SILENCE_MIN_RMS_DBFS = -60.0
SILENCE_FRAME_SECONDS = 0.1
SILENCE_LEADING_THRESHOLD_SECONDS = 0.2
SILENCE_TRAILING_THRESHOLD_SECONDS = 0.2
SILENCE_MIN_CONTENT_SECONDS = 1.0
SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}

VIEWPORT_JS_HELPER = """
const ViewportMath = (function() {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }
  function zoomViewport(viewport, zoomFactor, center, bounds, minSpan) {
    const xSpan = viewport.xMax - viewport.xMin;
    const ySpan = viewport.yMax - viewport.yMin;
    const newXSpan = Math.max(minSpan.x, xSpan * zoomFactor.x);
    const newYSpan = Math.max(minSpan.y, ySpan * zoomFactor.y);
    const xCenter = center.x;
    const yCenter = center.y;
    let xMin = xCenter - newXSpan / 2;
    let xMax = xCenter + newXSpan / 2;
    let yMin = yCenter - newYSpan / 2;
    let yMax = yCenter + newYSpan / 2;
    if (xMax - xMin > bounds.xMax - bounds.xMin) {
      xMin = bounds.xMin;
      xMax = bounds.xMax;
    } else {
      if (xMin < bounds.xMin) {
        const shift = bounds.xMin - xMin;
        xMin += shift;
        xMax += shift;
      }
      if (xMax > bounds.xMax) {
        const shift = xMax - bounds.xMax;
        xMin -= shift;
        xMax -= shift;
      }
    }
    if (yMax - yMin > bounds.yMax - bounds.yMin) {
      yMin = bounds.yMin;
      yMax = bounds.yMax;
    } else {
      if (yMin < bounds.yMin) {
        const shift = bounds.yMin - yMin;
        yMin += shift;
        yMax += shift;
      }
      if (yMax > bounds.yMax) {
        const shift = yMax - bounds.yMax;
        yMin -= shift;
        yMax -= shift;
      }
    }
    return { xMin, xMax, yMin, yMax };
  }
  function panViewport(viewport, delta, bounds) {
    let xMin = viewport.xMin + delta.x;
    let xMax = viewport.xMax + delta.x;
    let yMin = viewport.yMin + delta.y;
    let yMax = viewport.yMax + delta.y;
    const xSpan = xMax - xMin;
    const ySpan = yMax - yMin;
    if (xMin < bounds.xMin) {
      xMin = bounds.xMin;
      xMax = xMin + xSpan;
    }
    if (xMax > bounds.xMax) {
      xMax = bounds.xMax;
      xMin = xMax - xSpan;
    }
    if (yMin < bounds.yMin) {
      yMin = bounds.yMin;
      yMax = yMin + ySpan;
    }
    if (yMax > bounds.yMax) {
      yMax = bounds.yMax;
      yMin = yMax - ySpan;
    }
    return { xMin, xMax, yMin, yMax };
  }
  function viewportFromRect(start, end, bounds, minSpan) {
    let xMin = Math.min(start.x, end.x);
    let xMax = Math.max(start.x, end.x);
    let yMin = Math.min(start.y, end.y);
    let yMax = Math.max(start.y, end.y);
    if (xMax - xMin < minSpan.x) {
      const center = (xMin + xMax) / 2;
      xMin = center - minSpan.x / 2;
      xMax = center + minSpan.x / 2;
    }
    if (yMax - yMin < minSpan.y) {
      const center = (yMin + yMax) / 2;
      yMin = center - minSpan.y / 2;
      yMax = center + minSpan.y / 2;
    }
    xMin = clamp(xMin, bounds.xMin, bounds.xMax - minSpan.x);
    xMax = clamp(xMax, bounds.xMin + minSpan.x, bounds.xMax);
    yMin = clamp(yMin, bounds.yMin, bounds.yMax - minSpan.y);
    yMax = clamp(yMax, bounds.yMin + minSpan.y, bounds.yMax);
    return { xMin, xMax, yMin, yMax };
  }
  return { zoomViewport, panViewport, viewportFromRect };
})();
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ViewportMath };
}
"""


def _nice_num(value: float, round_value: bool) -> float:
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    fraction = value / (10**exp)
    if round_value:
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.0:
            nice = 2.0
        elif fraction < 7.0:
            nice = 5.0
        else:
            nice = 10.0
    else:
        if fraction <= 1.0:
            nice = 1.0
        elif fraction <= 2.0:
            nice = 2.0
        elif fraction <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
    return nice * (10**exp)


def _nice_linear_ticks(min_val: float, max_val: float, *, target: int = 6) -> list[float]:
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        return []
    if min_val == max_val:
        return [min_val]
    if target < 2:
        target = 2
    span = _nice_num(max_val - min_val, False)
    step = _nice_num(span / (target - 1), True)
    if step == 0:
        return [min_val, max_val]
    tick_min = math.ceil(min_val / step) * step
    tick_max = math.floor(max_val / step) * step
    ticks = []
    current = tick_min
    for _ in range(1000):
        if current > tick_max + step * 0.5:
            break
        if min_val <= current <= max_val:
            ticks.append(round(current, 10))
        current += step
    if not ticks:
        return [min_val, max_val]
    if ticks[0] > min_val:
        ticks.insert(0, round(min_val, 10))
    if ticks[-1] < max_val:
        ticks.append(round(max_val, 10))
    return ticks


def _nice_log_ticks(min_val: float, max_val: float) -> list[float]:
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        return []
    if min_val <= 0 or max_val <= 0 or min_val > max_val:
        return []
    decade_min = math.floor(math.log10(min_val))
    decade_max = math.ceil(math.log10(max_val))
    ticks = []
    for decade in range(decade_min, decade_max + 1):
        base = 10 ** decade
        for mult in (1, 2, 5):
            value = mult * base
            if min_val <= value <= max_val:
                ticks.append(float(value))
    return ticks


def _nice_ticks_py(
    min_val: float,
    max_val: float,
    *,
    log_scale: bool = False,
    target: int = 6
) -> list[float]:
    if log_scale:
        return _nice_log_ticks(min_val, max_val)
    return _nice_linear_ticks(min_val, max_val, target=target)


TICK_GENERATOR_JS = (
    "function niceNum(value,roundValue){"
    "  if(!(value>0)){return 0;}"
    "  const exp=Math.floor(Math.log10(value));"
    "  const fraction=value/Math.pow(10,exp);"
    "  let nice;"
    "  if(roundValue){"
    "    if(fraction<1.5){nice=1;}"
    "    else if(fraction<3){nice=2;}"
    "    else if(fraction<7){nice=5;}"
    "    else{nice=10;}"
    "  }else{"
    "    if(fraction<=1){nice=1;}"
    "    else if(fraction<=2){nice=2;}"
    "    else if(fraction<=5){nice=5;}"
    "    else{nice=10;}"
    "  }"
    "  return nice*Math.pow(10,exp);"
    "}"
    "function niceLinearTicks(minVal,maxVal,target){"
    "  if(!Number.isFinite(minVal)||!Number.isFinite(maxVal)){return [];}"
    "  if(minVal===maxVal){return [minVal];}"
    "  const count=Math.max(2,target||6);"
    "  const span=niceNum(maxVal-minVal,false);"
    "  const step=niceNum(span/(count-1),true);"
    "  if(step===0){return [minVal,maxVal];}"
    "  const tickMin=Math.ceil(minVal/step)*step;"
    "  const tickMax=Math.floor(maxVal/step)*step;"
    "  const ticks=[];"
    "  for(let v=tickMin,guard=0; guard<1000; guard++, v+=step){"
    "    if(v>tickMax+step*0.5){break;}"
    "    if(v>=minVal && v<=maxVal){"
    "      ticks.push(Math.round(v*1e10)/1e10);"
    "    }"
    "  }"
    "  if(!ticks.length){return [minVal,maxVal];}"
    "  if(ticks[0]>minVal){ticks.unshift(Math.round(minVal*1e10)/1e10);}"
    "  if(ticks[ticks.length-1]<maxVal){ticks.push(Math.round(maxVal*1e10)/1e10);}"
    "  return ticks;"
    "}"
    "function niceLogTicks(minVal,maxVal){"
    "  if(!Number.isFinite(minVal)||!Number.isFinite(maxVal)){return [];}"
    "  if(minVal<=0||maxVal<=0||minVal>maxVal){return [];}"
    "  const decadeMin=Math.floor(Math.log10(minVal));"
    "  const decadeMax=Math.ceil(Math.log10(maxVal));"
    "  const ticks=[];"
    "  for(let d=decadeMin; d<=decadeMax; d++){"
    "    const base=Math.pow(10,d);"
    "    [1,2,5].forEach((m)=>{"
    "      const value=m*base;"
    "      if(value>=minVal && value<=maxVal){ticks.push(value);}"
    "    });"
    "  }"
    "  return ticks;"
    "}"
    "function niceTicks(minVal,maxVal,options){"
    "  const opts=options||{};"
    "  if(opts.log){return niceLogTicks(minVal,maxVal);}"
    "  return niceLinearTicks(minVal,maxVal,opts.target||6);"
    "}"
    "function formatFrequency(value){"
    "  if(!Number.isFinite(value)){return 'n/a';}"
    "  if(value>=1000){"
    "    const k=value/1000;"
    "    const fixed=k>=100?0:(k>=10?1:2);"
    "    return `${k.toFixed(fixed)}k`;"
    "  }"
    "  if(value>=100){return value.toFixed(0);}"
    "  if(value>=10){return value.toFixed(1);}"
    "  return value.toFixed(2);"
    "}"
)


def _build_confidence(
    audio,
    *,
    effective_duration: float,
    silence_ratio: float,
    resampled: bool
) -> dict:
    """Build confidence assessment based on decode sanity checks."""
    reasons: list[str] = []
    if audio.samples.size == 0 or effective_duration <= 0:
        reasons.append("zero_length_audio")
    if effective_duration < MIN_EFFECTIVE_SECONDS:
        reasons.append(f"short_effective_duration<{MIN_EFFECTIVE_SECONDS}s")
    if silence_ratio >= 0.5:
        reasons.append("high_silence_ratio>=0.5")
    if any(
        "trimmed partial frame" in w or "decoded fewer frames" in w
        for w in audio.warnings
    ):
        reasons.append("truncated_decode")
    if resampled:
        reasons.append("resampled_audio")
    status = "pass" if not reasons else "warn"
    return {
        "status": status,
        "reasons": reasons,
        "downgraded": bool(reasons)
    }


def _compute_silence_ratio(
    samples: np.ndarray,
    fs: float,
    *,
    min_rms_dbfs: float = SILENCE_MIN_RMS_DBFS,
    frame_seconds: float = SILENCE_FRAME_SECONDS
) -> float:
    """Compute fraction of frames below RMS threshold."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("_compute_silence_ratio expects mono samples.")
    if x.size == 0:
        return 1.0
    frame_len = max(1, int(round(frame_seconds * fs)))
    total = 0
    silent = 0
    thresh = 10.0 ** (min_rms_dbfs / 20.0)
    for start in range(0, x.size, frame_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms <= thresh:
            silent += 1
        total += 1
    if total == 0:
        return 1.0
    return float(silent) / float(total)


def _resample_linear(samples: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Simple linear resampling for mono buffers."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x
    if fs <= 0 or target_fs <= 0:
        raise ValueError("Sample rates must be positive.")
    if fs == target_fs:
        return x
    n_out = max(1, int(round(x.size * target_fs / fs)))
    t_in = np.arange(x.size, dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / target_fs
    return np.interp(t_out, t_in, x).astype(np.float64)


def _iter_audio_files(folder: Path, recursive: bool) -> list[Path]:
    """Collect supported audio files from a folder."""
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder}")
    files: Iterable[Path]
    files = folder.rglob("*") if recursive else folder.glob("*")
    out: list[Path] = []
    for p in files:
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            out.append(p)
    return out


def _output_path(out_dir: Path, audio_path: Path) -> Path:
    """Build output report path for a given audio file."""
    safe_name = audio_path.stem + ".qcreport.json"
    return out_dir / safe_name


def _resolve_report_output_path(
    out_dir: Path | None,
    requested: str | None,
    default_name: str
) -> Path | None:
    """Resolve report output path anchored to the QC report directory."""
    if requested is None or out_dir is None:
        return None
    name = os.path.basename(requested) if requested else default_name
    if not name:
        name = default_name
    return out_dir / name


def _load_repair_plan(path: str) -> dict:
    """Load a repair plan from JSON or YAML."""
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Repair plan not found: {path}")
    raw = plan_path.read_text(encoding="utf-8")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        if importlib.util.find_spec("yaml") is None:
            raise ValueError("PyYAML is required for YAML repair plans.")
        yaml = importlib.import_module("yaml")
        try:
            plan = yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError(f"Failed to parse repair plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("Repair plan must be a mapping.")
    if "steps" not in plan:
        raise ValueError("Repair plan missing required 'steps' list.")
    return plan


def _batch_worker(
    args: tuple[str, str, str, str | None]
) -> tuple[str, str, str | None, dict | None]:
    """Worker for batch analysis."""
    audio_path, profile_path, mode, out_dir = args
    try:
        qcreport, decision, _, _ = _analyze_audio(audio_path, profile_path, mode=mode)
        if out_dir:
            out_path = _output_path(Path(out_dir), Path(audio_path))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(qcreport, indent=2), encoding="utf-8")
        return (audio_path, decision.overall_status.value, None, qcreport)
    except Exception as exc:
        return (audio_path, "error", str(exc), None)


def _render_markdown_summary(summary: dict) -> str:
    """Render a Markdown summary."""
    lines = []
    counts = summary.get("totals", {}).get("status_counts", {})
    lines.append("# Batch Summary")
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(f"- pass: {counts.get('pass', 0)}")
    lines.append(f"- warn: {counts.get('warn', 0)}")
    lines.append(f"- fail: {counts.get('fail', 0)}")
    lines.append(f"- error: {counts.get('error', 0)}")
    lines.append("")
    conf = summary.get("confidence", {}).get("counts", {})
    lines.append("## Confidence Counts")
    lines.append("")
    lines.append(f"- pass: {conf.get('pass', 0)}")
    lines.append(f"- warn: {conf.get('warn', 0)}")
    lines.append(f"- fail: {conf.get('fail', 0)}")
    lines.append("")
    lines.append("## Top Failure Causes")
    lines.append("")
    for cause, count in list(summary.get("failure_causes", {}).items())[:10]:
        lines.append(f"- {cause}: {count}")
    lines.append("")
    kpis = summary.get("kpis", {})
    lines.append("## Metric Distributions")
    lines.append("")
    for metric_key in ["tilt_deviation_db_per_oct", "true_peak_dbtp", "band_mean_deviation_db", "band_max_deviation_db", "band_variance_ratio"]:
        mean_val = kpis.get(f"mean_{metric_key}")
        if mean_val is not None:
            lines.append(f"- {metric_key}: mean={mean_val:.3f}")
    lines.append("")
    return "\n".join(lines)


def _render_corpus_report_md(summary: dict, *, profile_path: str, mode: str) -> str:
    """Render a one-page Markdown report for batch results."""
    counts = summary.get("totals", {}).get("status_counts", {})
    conf = summary.get("confidence", {}).get("counts", {})
    causes = list(summary.get("failure_causes", {}).items())
    conf_reasons = list(summary.get("confidence", {}).get("top_reasons", {}).items())
    kpis = summary.get("kpis", {})

    lines = []
    lines.append("# SpectraQC Batch Report")
    lines.append("")
    lines.append(f"- Profile: `{profile_path}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append("")
    lines.append("## Corpus Results")
    lines.append("")
    lines.append(f"- pass: {counts.get('pass', 0)}")
    lines.append(f"- warn: {counts.get('warn', 0)}")
    lines.append(f"- fail: {counts.get('fail', 0)}")
    lines.append(f"- error: {counts.get('error', 0)}")
    lines.append("")
    lines.append("## Confidence Notes")
    lines.append("")
    lines.append(f"- pass: {conf.get('pass', 0)}")
    lines.append(f"- warn: {conf.get('warn', 0)}")
    lines.append(f"- fail: {conf.get('fail', 0)}")
    if conf_reasons:
        lines.append("")
        lines.append("Top confidence reasons:")
        for reason, count in conf_reasons[:5]:
            lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("## KPI Metrics")
    lines.append("")
    for metric_key in ["tilt_deviation_db_per_oct", "true_peak_dbtp", "band_mean_deviation_db", "band_max_deviation_db", "band_variance_ratio"]:
        mean_val = kpis.get(f"mean_{metric_key}")
        if mean_val is not None:
            lines.append(f"- mean_{metric_key}: {mean_val:.3f}")
    lines.append("")
    lines.append("## Notable Failures")
    lines.append("")
    if causes:
        for cause, count in causes[:10]:
            lines.append(f"- {cause}: {count}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _render_corpus_report_html(
    summary: dict,
    *,
    profile_path: str,
    mode: str,
    file_links: list[tuple[str, str, str]] | None = None,
    embedded_reports: dict[str, dict] | None = None
) -> str:
    """Render a one-page HTML report for batch results."""
    counts = summary.get("totals", {}).get("status_counts", {})
    conf = summary.get("confidence", {}).get("counts", {})
    causes = list(summary.get("failure_causes", {}).items())
    conf_reasons = list(summary.get("confidence", {}).get("top_reasons", {}).items())
    kpis = summary.get("kpis", {})

    def _svg_bar(name: str, value: int, max_value: int, color: str) -> str:
        width = 240
        height = 18
        fill_w = int(width * (value / max_value)) if max_value > 0 else 0
        return (
            f"<div class='bar-row'><span class='bar-label'>{name}</span>"
            f"<svg width='{width}' height='{height}' role='img' aria-label='{name}'>"
            f"<rect width='{width}' height='{height}' fill='#f0f0f0' />"
            f"<rect width='{fill_w}' height='{height}' fill='{color}' />"
            f"</svg><span class='bar-value'>{value}</span></div>"
        )

    total = max(1, sum(counts.values()))
    conf_total = max(1, sum(conf.values()))

    kpi_rows = []
    for metric_key in ["tilt_deviation_db_per_oct", "true_peak_dbtp", "band_mean_deviation_db", "band_max_deviation_db", "band_variance_ratio"]:
        mean_val = kpis.get(f"mean_{metric_key}")
        if mean_val is not None:
            kpi_rows.append(f"<tr><td>{metric_key}</td><td>{mean_val:.3f}</td></tr>")
    
    kpi_table = (
        "<table><thead><tr>"
        "<th>Metric</th><th>Mean</th>"
        "</tr></thead><tbody>"
        + "".join(kpi_rows) +
        "</tbody></table>"
    ) if kpi_rows else "<p>No KPI data available</p>"

    top_causes = "".join(
        f"<li>{cause}: {count}</li>" for cause, count in causes[:10]
    ) or "<li>None</li>"
    top_conf = "".join(
        f"<li>{reason}: {count}</li>" for reason, count in conf_reasons[:5]
    ) or "<li>None</li>"

    status_bars = "".join([
        _svg_bar("pass", counts.get("pass", 0), total, "#4caf50"),
        _svg_bar("warn", counts.get("warn", 0), total, "#ffb300"),
        _svg_bar("fail", counts.get("fail", 0), total, "#e53935"),
        _svg_bar("error", counts.get("error", 0), total, "#6d6d6d"),
    ])
    conf_bars = "".join([
        _svg_bar("pass", conf.get("pass", 0), conf_total, "#4caf50"),
        _svg_bar("warn", conf.get("warn", 0), conf_total, "#ffb300"),
        _svg_bar("fail", conf.get("fail", 0), conf_total, "#e53935"),
    ])

    embedded_reports = embedded_reports or {}
    embedded_reports_json = json.dumps(embedded_reports, ensure_ascii=True).replace("</", "<\\/")
    viewer_section_html, viewer_section_css, viewer_section_js = _render_qcreport_viewer_section(
        file_links,
        embedded_reports_json
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SpectraQC Batch Report</title>"
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}"
        "h1,h2{margin:0 0 12px 0}"
        ".meta{color:#555;margin-bottom:16px}"
        ".section{margin:20px 0}"
        ".bar-row{display:flex;align-items:center;gap:8px;margin:6px 0}"
        ".bar-label{width:60px;text-transform:uppercase;font-size:12px;color:#555}"
        ".bar-value{width:40px;text-align:right;font-variant-numeric:tabular-nums}"
        "table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "td.file{max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        "td.status{font-weight:bold;text-transform:uppercase}"
        "td.status.pass{color:#2e7d32}"
        "td.status.warn{color:#f9a825}"
        "td.status.fail{color:#c62828}"
        "td.status.error{color:#6d6d6d}"
        "ul{margin:6px 0 0 18px}"
        f"{viewer_section_css}"
        "</style></head><body>"
        "<h1>SpectraQC Batch Report</h1>"
        f"<div class='meta'>Profile: <code>{profile_path}</code> | Mode: <code>{mode}</code></div>"
        "<div class='section'><h2>Corpus Results</h2>"
        f"{status_bars}</div>"
        "<div class='section'><h2>Confidence Notes</h2>"
        f"{conf_bars}"
        "<h3>Top confidence reasons</h3><ul>"
        f"{top_conf}</ul></div>"
        "<div class='section'><h2>KPI Metrics</h2>"
        f"{kpi_table}</div>"
        "<div class='section'><h2>Notable Failures</h2><ul>"
        f"{top_causes}</ul></div>"
        f"{viewer_section_html}"
        f"{viewer_section_js}"
        "</body></html>"
    )


def _render_qcreport_viewer_section(
    file_links: list[tuple[str, str, str]] | None,
    embedded_reports_json: str
) -> tuple[str, str, str]:
    """Render an embedded GUI viewer for QCReport JSON files."""
    file_input_class = " viewer-hidden" if file_links else ""
    batch_reports_json = "[]"
    if file_links:
        batch_reports = [{"label": label, "href": href} for label, _, href in file_links]
        batch_reports_json = json.dumps(batch_reports, ensure_ascii=True).replace("</", "<\\/")

    viewer_html = (
        "<div class='section' id='viewer'>"
        "<h2>Report Viewer</h2>"
        "<div class='viewer-row' style='margin-top:16px'>"
        "<div class='viewer-panel'><h3>Reports</h3>"
        f"<div id='viewer-file-input' class='viewer-file-input{file_input_class}'>"
        "<p>Select one or more <code>.qcreport.json</code> files.</p>"
        "<input id='viewer-files' type='file' multiple accept='.json'>"
        "</div>"
        "<table id='viewer-list'><thead><tr>"
        "<th>File</th><th>Status</th><th>Confidence</th>"
        "</tr></thead><tbody></tbody></table>"
        "</div>"
        "<div class='viewer-panel'><h3>Details</h3>"
        "<div id='viewer-details'>Select a report.</div>"
        "<div id='viewer-charts'></div>"
        "</div>"
        "</div>"
        "<div id='viewer-overlay' class='viewer-overlay' aria-hidden='true'>"
        "<div class='viewer-overlay-inner'>"
        "<div class='viewer-overlay-header'>"
        "<h3 id='viewer-overlay-title'>Chart</h3>"
        "<button id='viewer-overlay-close' class='viewer-overlay-close' type='button'>Close</button>"
        "</div>"
        "<div id='viewer-overlay-body' class='viewer-overlay-body'></div>"
        "</div>"
        "</div>"
        "</div>"
    )
    viewer_css = (
        ".viewer-row{display:flex;gap:16px;flex-wrap:wrap}"
        ".viewer-panel{flex:1;min-width:320px;border:1px solid #eee;padding:12px;border-radius:8px}"
        ".viewer-file-input{margin-bottom:12px}"
        ".viewer-file-input.viewer-hidden{display:none}"
        "#viewer-list{border-collapse:collapse;width:100%;font-size:14px}"
        "#viewer-list th,#viewer-list td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}"
        "#viewer-charts{margin-top:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}"
        ".viewer-chart{border:1px solid #f0f0f0;padding:8px;border-radius:6px}"
        ".viewer-chart h3{margin:4px 0 8px 0;font-size:14px}"
        ".viewer-chart canvas{width:100%;height:auto}"
        ".viewer-expand{margin-top:6px;border:1px solid #ddd;background:#fafafa;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px}"
        ".viewer-expand:hover{background:#f0f0f0}"
        ".viewer-note{color:#555;margin-top:6px;font-size:12px}"
        ".viewer-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.75);display:none;align-items:center;justify-content:center;z-index:9999}"
        ".viewer-overlay.active{display:flex}"
        ".viewer-overlay-inner{background:#fff;border-radius:10px;max-width:96vw;max-height:92vh;width:100%;padding:12px;box-shadow:0 10px 40px rgba(0,0,0,0.35);display:flex;flex-direction:column}"
        ".viewer-overlay-header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}"
        ".viewer-overlay-header h3{margin:0;font-size:16px}"
        ".viewer-overlay-close{border:1px solid #ddd;background:#fafafa;border-radius:4px;padding:6px 10px;cursor:pointer;font-size:12px}"
        ".viewer-overlay-close:hover{background:#f0f0f0}"
        ".viewer-overlay-body{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto}"
        ".viewer-overlay-body img{max-width:100%;max-height:100%;height:auto}"
    )
    viewer_js = (
        "<script>"
        + TICK_GENERATOR_JS +
        "const filesInput=document.getElementById('viewer-files');"
        "const fileInputWrap=document.getElementById('viewer-file-input');"
        "const listBody=document.querySelector('#viewer-list tbody');"
        "const details=document.getElementById('viewer-details');"
        "const charts=document.getElementById('viewer-charts');"
        "const overlay=document.getElementById('viewer-overlay');"
        "const overlayBody=document.getElementById('viewer-overlay-body');"
        "const overlayTitle=document.getElementById('viewer-overlay-title');"
        "const overlayClose=document.getElementById('viewer-overlay-close');"
        "const reportCache=new Map();"
        f"const batchReports={batch_reports_json};"
        f"const embeddedReports={embedded_reports_json};"
        "function closeOverlay(){"
        "  overlay.classList.remove('active');"
        "  overlay.setAttribute('aria-hidden','true');"
        "  overlayBody.innerHTML='';"
        "}"
        "function openOverlayFromCanvas(canvas,title){"
        "  const img=document.createElement('img');"
        "  img.src=canvas.toDataURL('image/png');"
        "  overlayBody.innerHTML='';"
        "  overlayBody.appendChild(img);"
        "  overlayTitle.textContent=title||'Chart';"
        "  overlay.classList.add('active');"
        "  overlay.setAttribute('aria-hidden','false');"
        "}"
        "overlayClose.addEventListener('click',closeOverlay);"
        "overlay.addEventListener('click',(e)=>{"
        "  if(e.target===overlay){closeOverlay();}"
        "});"
        "document.addEventListener('keydown',(e)=>{"
        "  if(e.key==='Escape'){closeOverlay();}"
        "});"
        "function escapeHtml(value){"
        "  const map={'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'};"
        "  return String(value).replace(/[&<>\"']/g,(c)=>map[c]);"
        "}"
        "function formatLabel(label){"
        "  const cleaned=String(label).replace(/_/g,' ');"
        "  return cleaned.length>16?cleaned.slice(0,15)+'...':cleaned;"
        "}"
        "function formatNumber(value){"
        "  if(!Number.isFinite(value)) return 'n/a';"
        "  return Math.abs(value)>=10?value.toFixed(1):value.toFixed(2);"
        "}"
        "function appendLoadedRow(label,status,conf,report){"
        "  const tr=document.createElement('tr');"
        "  tr.innerHTML=`<td>${escapeHtml(label)}</td><td class='status ${status}'>${status}</td><td>${conf}</td>`;"
        "  tr.addEventListener('click',()=>renderDetails(report,label));"
        "  listBody.appendChild(tr);"
        "}"
        "filesInput.addEventListener('change', async (e)=>{"
        "  listBody.innerHTML=''; details.textContent='Select a report.'; charts.innerHTML='';"
        "  const files=[...e.target.files];"
        "  for(const f of files){"
        "    const text=await f.text();"
        "    try{"
        "      const j=JSON.parse(text);"
        "      const status=j.decisions?.overall_status||'unknown';"
        "      const conf=j.confidence?.status||'unknown';"
        "      reportCache.set(f.name,j);"
        "      appendLoadedRow(f.name,status,conf,j);"
        "    }catch(err){"
        "      const tr=document.createElement('tr');"
        "      tr.innerHTML=`<td>${escapeHtml(f.name)}</td><td class='status error'>error</td><td>invalid json</td>`;"
        "      listBody.appendChild(tr);"
        "    }"
        "  }"
        "});"
        "async function loadReportFromUrl(url,label){"
        "  if(embeddedReports && embeddedReports[label]){"
        "    const j=embeddedReports[label];"
        "    const status=j.decisions?.overall_status||'unknown';"
        "    const conf=j.confidence?.status||'unknown';"
        "    reportCache.set(label,j);"
        "    appendLoadedRow(label,status,conf,j);"
        "    return;"
        "  }"
        "  try{"
        "    const res=await fetch(url);"
        "    if(!res.ok){throw new Error(`HTTP ${res.status}`);}"
        "    const j=await res.json();"
        "    const status=j.decisions?.overall_status||'unknown';"
        "    const conf=j.confidence?.status||'unknown';"
        "    reportCache.set(label,j);"
        "    appendLoadedRow(label,status,conf,j);"
        "  }catch(err){"
        "    if(fileInputWrap){fileInputWrap.classList.remove('viewer-hidden');}"
        "  }"
        "}"
        "async function loadAllBatchReports(){"
        "  if(!batchReports.length){return;}"
        "  const tasks=batchReports.map((item)=>loadReportFromUrl(item.href,item.label));"
        "  await Promise.allSettled(tasks);"
        "  const first=listBody.querySelector('tr');"
        "  if(first){first.click();}"
        "}"
        "function renderBarChart(container,title,labels,values,units){"
        "  if(!labels.length||!values.length) return;"
        "  const wrapper=document.createElement('div');"
        "  wrapper.className='viewer-chart';"
        "  const h=document.createElement('h3');"
        "  h.textContent=title;"
        "  wrapper.appendChild(h);"
        "  const canvas=document.createElement('canvas');"
        "  canvas.width=560; canvas.height=260;"
        "  wrapper.appendChild(canvas);"
        "  const ctx=canvas.getContext('2d');"
        "  const padding={top:24,right:16,bottom:58,left:64};"
        "  const w=canvas.width-padding.left-padding.right;"
        "  const hgt=canvas.height-padding.top-padding.bottom;"
        "  const minVal=Math.min(0,...values);"
        "  const maxVal=Math.max(0,...values);"
        "  const range=(maxVal-minVal)||1;"
        "  const scaleY=hgt/range;"
        "  const yZero=padding.top+(maxVal-0)*scaleY;"
        "  ctx.clearRect(0,0,canvas.width,canvas.height);"
        "  ctx.strokeStyle='#ccc'; ctx.lineWidth=1;"
        "  ctx.beginPath(); ctx.moveTo(padding.left,yZero); ctx.lineTo(padding.left+w,yZero); ctx.stroke();"
        "  ctx.beginPath(); ctx.moveTo(padding.left,padding.top); ctx.lineTo(padding.left,padding.top+hgt); ctx.stroke();"
        "  const yTicks=5;"
        "  ctx.fillStyle='#666'; ctx.font='12px Arial,Helvetica,sans-serif';"
        "  for(let i=0;i<=yTicks;i++){"
        "    const t=minVal+(i*(range/yTicks));"
        "    const y=padding.top+((maxVal-t)/range)*hgt;"
        "    ctx.strokeStyle='#e0e0e0'; ctx.beginPath(); ctx.moveTo(padding.left,y); ctx.lineTo(padding.left+w,y); ctx.stroke();"
        "    ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(padding.left-4,y); ctx.lineTo(padding.left,y); ctx.stroke();"
        "    const label=formatNumber(t)+(units?` ${units}`:'');"
        "    ctx.fillText(label,4,y+4);"
        "  }"
        "  const barWidth=w/values.length;"
        "  const labelStep=Math.max(1,Math.ceil(values.length/10));"
        "  for(let i=0;i<values.length;i++){"
        "    const v=values[i];"
        "    const x=padding.left+i*barWidth+4;"
        "    const barH=Math.abs(v)*scaleY;"
        "    const y=v>=0?yZero-barH:yZero;"
        "    ctx.fillStyle=v>=0?'#4caf50':'#e53935';"
        "    ctx.fillRect(x,y,Math.max(2,barWidth-8),barH);"
        "    if(i%labelStep===0){"
        "      ctx.save();"
        "      ctx.translate(x, padding.top+hgt+12);"
        "      ctx.rotate(-0.35);"
        "      ctx.fillStyle='#555';"
        "      ctx.fillText(formatLabel(labels[i]),0,0);"
        "      ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(0,-8); ctx.lineTo(0,-2); ctx.stroke();"
        "      ctx.restore();"
        "    }"
        "  }"
        "  const xLabel='Band';"
        "  const yLabel=units?`Value (${units})`:'Value';"
        "  ctx.fillStyle='#555'; ctx.font='13px Arial,Helvetica,sans-serif';"
        "  ctx.textAlign='center';"
        "  ctx.fillText(xLabel,padding.left+(w/2),canvas.height-10);"
        "  ctx.save();"
        "  ctx.translate(16,padding.top+(hgt/2));"
        "  ctx.rotate(-Math.PI/2);"
        "  ctx.fillText(yLabel,0,0);"
        "  ctx.restore();"
        "  ctx.textAlign='left';"
        "  const note=document.createElement('div');"
        "  note.className='viewer-note';"
        "  note.textContent=`min ${formatNumber(minVal)}${units?` ${units}`:''}, max ${formatNumber(maxVal)}${units?` ${units}`:''}`;"
        "  const expand=document.createElement('button');"
        "  expand.className='viewer-expand';"
        "  expand.type='button';"
        "  expand.textContent='Full screen';"
        "  expand.addEventListener('click',()=>openOverlayFromCanvas(canvas,title));"
        "  wrapper.appendChild(note);"
        "  wrapper.appendChild(expand);"
        "  container.appendChild(wrapper);"
        "}"
        "function renderMultiLineChart(container,title,xs,series,units,useLog){"
        "  if(!xs.length||!series.length) return;"
        "  const wrapper=document.createElement('div');"
        "  wrapper.className='viewer-chart';"
        "  const h=document.createElement('h3');"
        "  h.textContent=title;"
        "  wrapper.appendChild(h);"
        "  const canvas=document.createElement('canvas');"
        "  canvas.width=560; canvas.height=260;"
        "  wrapper.appendChild(canvas);"
        "  const ctx=canvas.getContext('2d');"
        "  const padding={top:24,right:16,bottom:56,left:64};"
        "  const w=canvas.width-padding.left-padding.right;"
        "  const hgt=canvas.height-padding.top-padding.bottom;"
        "  const allValues=series.flatMap(s=>s.values||[]);"
        "  if(!allValues.length) return;"
        "  const minY=Math.min(...allValues);"
        "  const maxY=Math.max(...allValues);"
        "  const yRange=(maxY-minY)||1;"
        "  const xsLog=useLog?xs.map(x=>Math.log10(x)):xs;"
        "  const minX=Math.min(...xsLog);"
        "  const maxX=Math.max(...xsLog);"
        "  const xRange=(maxX-minX)||1;"
        "  ctx.clearRect(0,0,canvas.width,canvas.height);"
        "  ctx.strokeStyle='#ccc'; ctx.lineWidth=1;"
        "  ctx.beginPath(); ctx.moveTo(padding.left,padding.top); ctx.lineTo(padding.left,padding.top+hgt); ctx.stroke();"
        "  ctx.beginPath(); ctx.moveTo(padding.left,padding.top+hgt); ctx.lineTo(padding.left+w,padding.top+hgt); ctx.stroke();"
        "  const yTicks=niceTicks(minY,maxY,{log:false,target:6});"
        "  ctx.fillStyle='#666'; ctx.font='12px Arial,Helvetica,sans-serif';"
        "  yTicks.forEach((t)=>{"
        "    const y=padding.top+((maxY-t)/yRange)*hgt;"
        "    ctx.strokeStyle='#e0e0e0'; ctx.beginPath(); ctx.moveTo(padding.left,y); ctx.lineTo(padding.left+w,y); ctx.stroke();"
        "    ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(padding.left-4,y); ctx.lineTo(padding.left,y); ctx.stroke();"
        "    const label=formatNumber(t)+(units?` ${units}`:'');"
        "    ctx.fillText(label,4,y+4);"
        "  });"
        "  series.forEach((s,idx)=>{"
        "    const color=s.color||['#1e88e5','#8e24aa','#43a047'][idx%3];"
        "    ctx.strokeStyle=color; ctx.lineWidth=1.5;"
        "    ctx.beginPath();"
        "    s.values.forEach((v,i)=>{"
        "      const xVal=xsLog[i];"
        "      if(!Number.isFinite(v)||!Number.isFinite(xVal)) return;"
        "      const x=padding.left+((xVal-minX)/xRange)*w;"
        "      const y=padding.top+((maxY-v)/yRange)*hgt;"
        "      if(i===0){ctx.moveTo(x,y);}else{ctx.lineTo(x,y);}"
        "    });"
        "    ctx.stroke();"
        "  });"
        "  const xAxisLabel=useLog?'Frequency (Hz, log scale)':'Frequency (Hz)';"
        "  const yAxisLabel=units?`Value (${units})`:'Value';"
        "  ctx.fillStyle='#555'; ctx.font='13px Arial,Helvetica,sans-serif';"
        "  ctx.textAlign='center';"
        "  ctx.fillText(xAxisLabel,padding.left+(w/2),canvas.height-10);"
        "  ctx.save();"
        "  ctx.translate(16,padding.top+(hgt/2));"
        "  ctx.rotate(-Math.PI/2);"
        "  ctx.fillText(yAxisLabel,0,0);"
        "  ctx.restore();"
        "  const legend=document.createElement('div');"
        "  legend.className='viewer-note';"
        "  legend.innerHTML=series.map((s,idx)=>{"
        "    const color=s.color||['#1e88e5','#8e24aa','#43a047'][idx%3];"
        "    return `<span style=\"display:inline-flex;align-items:center;margin-right:8px;\">"
        "      <span style=\"width:10px;height:10px;background:${color};display:inline-block;margin-right:4px;border-radius:2px;\"></span>"
        "      ${escapeHtml(s.label||'series')}</span>`;"
        "  }).join('');"
        "  const expand=document.createElement('button');"
        "  expand.className='viewer-expand';"
        "  expand.type='button';"
        "  expand.textContent='Full screen';"
        "  expand.addEventListener('click',()=>openOverlayFromCanvas(canvas,title));"
        "  wrapper.appendChild(legend);"
        "  wrapper.appendChild(expand);"
        "  container.appendChild(wrapper);"
        "}"
        "function renderLineChart(container,title,xs,ys,units,useLog){"
        "  const points=[];"
        "  const n=Math.min(xs.length,ys.length);"
        "  for(let i=0;i<n;i++){"
        "    const x=xs[i];"
        "    const y=ys[i];"
        "    if(!Number.isFinite(x)||!Number.isFinite(y)) continue;"
        "    if(useLog && x<=0) continue;"
        "    points.push([x,y]);"
        "  }"
        "  if(!points.length) return;"
        "  const wrapper=document.createElement('div');"
        "  wrapper.className='viewer-chart';"
        "  const h=document.createElement('h3');"
        "  h.textContent=title;"
        "  wrapper.appendChild(h);"
        "  const canvas=document.createElement('canvas');"
        "  canvas.width=560; canvas.height=260;"
        "  wrapper.appendChild(canvas);"
        "  const ctx=canvas.getContext('2d');"
        "  const padding={top:24,right:16,bottom:56,left:64};"
        "  const w=canvas.width-padding.left-padding.right;"
        "  const hgt=canvas.height-padding.top-padding.bottom;"
        "  const xsVals=points.map(p=>p[0]);"
        "  const ysVals=points.map(p=>p[1]);"
        "  const minY=Math.min(...ysVals);"
        "  const maxY=Math.max(...ysVals);"
        "  const yRange=(maxY-minY)||1;"
        "  const minX=Math.min(...xsVals);"
        "  const maxX=Math.max(...xsVals);"
        "  const xMin=useLog?Math.log10(minX):minX;"
        "  const xMax=useLog?Math.log10(maxX):maxX;"
        "  const xRange=(xMax-xMin)||1;"
        "  ctx.clearRect(0,0,canvas.width,canvas.height);"
        "  ctx.strokeStyle='#ccc'; ctx.lineWidth=1;"
        "  ctx.beginPath(); ctx.moveTo(padding.left,padding.top); ctx.lineTo(padding.left,padding.top+hgt); ctx.stroke();"
        "  ctx.beginPath(); ctx.moveTo(padding.left,padding.top+hgt); ctx.lineTo(padding.left+w,padding.top+hgt); ctx.stroke();"
        "  ctx.strokeStyle='#1f77b4'; ctx.lineWidth=1.5;"
        "  ctx.beginPath();"
        "  points.forEach((p,idx)=>{"
        "    const rawX=useLog?Math.log10(p[0]):p[0];"
        "    const x=padding.left+((rawX-xMin)/xRange)*w;"
        "    const y=padding.top+((maxY-p[1])/yRange)*hgt;"
        "    if(idx===0){ctx.moveTo(x,y);}else{ctx.lineTo(x,y);}"
        "  });"
        "  ctx.stroke();"
        "  const yTicks=niceTicks(minY,maxY,{log:false,target:6});"
        "  ctx.fillStyle='#666'; ctx.font='12px Arial,Helvetica,sans-serif';"
        "  yTicks.forEach((t)=>{"
        "    const y=padding.top+((maxY-t)/yRange)*hgt;"
        "    ctx.strokeStyle='#e0e0e0'; ctx.beginPath(); ctx.moveTo(padding.left,y); ctx.lineTo(padding.left+w,y); ctx.stroke();"
        "    ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(padding.left-4,y); ctx.lineTo(padding.left,y); ctx.stroke();"
        "    const label=formatNumber(t)+(units?` ${units}`:'');"
        "    ctx.fillText(label,4,y+4);"
        "  });"
        "  const xTickVals=niceTicks(minX,maxX,{log:useLog,target:6});"
        "  const maxLabels=Math.max(3,Math.floor(w/70));"
        "  const labelStep=Math.max(1,Math.ceil(xTickVals.length/maxLabels));"
        "  const rotation=labelStep>1?-0.6:-0.35;"
        "  ctx.fillStyle='#666'; ctx.font='12px Arial,Helvetica,sans-serif';"
        "  xTickVals.forEach((xVal,idx)=>{"
        "    if(idx%labelStep!==0){return;}"
        "    const raw=useLog?Math.log10(xVal):xVal;"
        "    const x=padding.left+((raw-xMin)/xRange)*w;"
        "    ctx.strokeStyle='#888'; ctx.beginPath(); ctx.moveTo(x,padding.top+hgt); ctx.lineTo(x,padding.top+hgt+6); ctx.stroke();"
        "    const label=formatFrequency(xVal);"
        "    ctx.save();"
        "    ctx.translate(x,padding.top+hgt+18);"
        "    ctx.rotate(rotation);"
        "    ctx.textAlign='center';"
        "    ctx.fillText(label,0,0);"
        "    ctx.restore();"
        "  });"
        "  const xAxisLabel=useLog?'Frequency (Hz, log scale)':'Frequency (Hz)';"
        "  const yAxisLabel=units?`Value (${units})`:'Value';"
        "  ctx.fillStyle='#555'; ctx.font='13px Arial,Helvetica,sans-serif';"
        "  ctx.textAlign='center';"
        "  ctx.fillText(xAxisLabel,padding.left+(w/2),canvas.height-10);"
        "  ctx.save();"
        "  ctx.translate(16,padding.top+(hgt/2));"
        "  ctx.rotate(-Math.PI/2);"
        "  ctx.fillText(yAxisLabel,0,0);"
        "  ctx.restore();"
        "  ctx.textAlign='left';"
        "  const note=document.createElement('div');"
        "  note.className='viewer-note';"
        "  const xLabel=useLog?'Hz (log scale)':'Hz';"
        "  note.textContent=`${xLabel}, min ${formatNumber(minY)}${units?` ${units}`:''}, max ${formatNumber(maxY)}${units?` ${units}`:''}`;"
        "  const expand=document.createElement('button');"
        "  expand.className='viewer-expand';"
        "  expand.type='button';"
        "  expand.textContent='Full screen';"
        "  expand.addEventListener('click',()=>openOverlayFromCanvas(canvas,title));"
        "  wrapper.appendChild(note);"
        "  wrapper.appendChild(expand);"
        "  container.appendChild(wrapper);"
        "}"
        "function renderDetails(j,name){"
        "  const status=j.decisions?.overall_status||'unknown';"
        "  const conf=j.confidence?.status||'unknown';"
        "  const inputPath=j.input?.path||'';"
        "  const duration=j.input?.duration_s;"
        "  const durationText=Number.isFinite(duration)?`${duration.toFixed(2)} s`:'n/a';"
        "  const notes=[];"
        "  const bands=j.decisions?.band_decisions||[];"
        "  for(const bd of bands){"
        "    for(const key of ['mean','max','variance']){"
        "      const d=bd[key];"
        "      if(d && (d.status==='warn' || d.status==='fail')){"
        "        if(d.notes) notes.push(d.notes);"
        "      }"
        "    }"
        "  }"
        "  const globals=j.decisions?.global_decisions||[];"
        "  for(const gd of globals){"
        "    if(gd && (gd.status==='warn' || gd.status==='fail')){"
        "      if(gd.notes) notes.push(gd.notes);"
        "    }"
        "  }"
        "  details.innerHTML = `"
        "    <strong>${escapeHtml(name)}</strong><br>"
        "    Status: <span class='status ${status}'>${status}</span><br>"
        "    Confidence: ${conf}<br>"
        "    Duration: ${durationText}<br>"
        "    <span class='viewer-note'>${escapeHtml(inputPath)}</span><br><br>"
        "    <strong>Notable notes</strong><ul>${notes.slice(0,10).map(n=>`<li>${escapeHtml(n)}</li>`).join('')||'<li>None</li>'}</ul>"
        "  `;"
        "  charts.innerHTML='';"
        "  const metrics=j.metrics||{};"
        "  const grid=metrics.frequency_grid?.freqs_hz||[];"
        "  const ltpsd=metrics.ltpsd?.mean_db||[];"
        "  const deviation=metrics.deviation?.delta_mean_db||[];"
        "  const reference=metrics.reference||{};"
        "  const refMean=reference.mean_db||[];"
        "  const refVar=reference.var_db2||[];"
        "  if(grid.length && ltpsd.length && refMean.length){"
        "    renderMultiLineChart(charts,'Reference vs Input Mean',grid,["
        "      {label:'Input',values:ltpsd,color:'#1e88e5'},"
        "      {label:'Reference',values:refMean,color:'#8e24aa'}"
        "    ],'dB',true);"
        "  }"
        "  if(grid.length && ltpsd.length && !refMean.length){"
        "    renderLineChart(charts,'LTPSD Mean',grid,ltpsd,'dB',true);"
        "  }"
        "  if(grid.length && refVar.length){"
        "    renderLineChart(charts,'Reference Variance',grid,refVar,'dB²',true);"
        "  }"
        "  if(grid.length && deviation.length){"
        "    renderLineChart(charts,'Deviation Curve',grid,deviation,'dB',true);"
        "  }"
        "  const bandMetrics=metrics.band_metrics||[];"
        "  const meanLabels=[]; const meanValues=[];"
        "  const maxLabels=[]; const maxValues=[];"
        "  const varLabels=[]; const varValues=[];"
        "  for(const bm of bandMetrics){"
        "    const label=bm.band_name||'band';"
        "    if(Number.isFinite(bm.mean_deviation_db)){meanLabels.push(label); meanValues.push(bm.mean_deviation_db);}"
        "    if(Number.isFinite(bm.max_deviation_db)){maxLabels.push(label); maxValues.push(bm.max_deviation_db);}"
        "    if(Number.isFinite(bm.variance_ratio)){varLabels.push(label); varValues.push(bm.variance_ratio);}"
        "  }"
        "  if(meanValues.length){renderBarChart(charts,'Band Mean Deviation',meanLabels,meanValues,'dB');}"
        "  if(maxValues.length){renderBarChart(charts,'Band Max Deviation',maxLabels,maxValues,'dB');}"
        "  if(varValues.length){renderBarChart(charts,'Band Variance Ratio',varLabels,varValues,'');}"
        "  const gm=metrics.global_metrics||{};"
        "  const gLabels=[]; const gValues=[];"
        "  const gUnitsMap={spectral_tilt_db_per_oct:'dB/oct',tilt_deviation_db_per_oct:'dB/oct',true_peak_dbtp:'dBTP'};"
        "  for(const [key,val] of Object.entries(gm)){"
        "    if(Number.isFinite(val)){gLabels.push(key); gValues.push(val);}"
        "  }"
        "  if(gValues.length){"
        "    const units=gLabels.length===1?gUnitsMap[gLabels[0]]||'':'';"
        "    renderBarChart(charts,'Global Metrics',gLabels,gValues,units);"
        "  }"
        "}"
        "loadAllBatchReports();"
        "</script>"
    )
    return viewer_html, viewer_css, viewer_js


def _render_repro_doc(
    *,
    profile,
    algorithm_ids: list[str],
    profile_path: str,
    mode: str,
    audio_path: str | None = None,
    manifest_path: str | None = None,
    folder_path: str | None = None,
    recursive: bool = False
) -> str:
    """Render reproducibility documentation."""
    lines = []
    lines.append("# Reproducibility")
    lines.append("")
    lines.append("## Tooling")
    lines.append("")
    lines.append(f"- spectraqc_version: `{__version__}`")
    lines.append("")
    lines.append("## Profile")
    lines.append("")
    lines.append(f"- profile_path: `{profile_path}`")
    lines.append(f"- profile_hash_sha256: `{profile.profile_hash_sha256}`")
    lines.append(f"- profile_version: `{profile.version}`")
    lines.append("")
    lines.append("## Algorithms")
    lines.append("")
    for algo_id in algorithm_ids:
        lines.append(f"- {algo_id}")
    lines.append("")
    lines.append("## Rerun Instructions")
    lines.append("")
    if audio_path:
        lines.append("```bash")
        lines.append(f"spectraqc analyze \"{audio_path}\" --profile \"{profile_path}\" --mode {mode}")
        lines.append("```")
    elif manifest_path or folder_path:
        lines.append("```bash")
        if manifest_path:
            lines.append(
                f"spectraqc batch --manifest \"{manifest_path}\" --profile \"{profile_path}\" --mode {mode}"
            )
        if folder_path:
            rec = " --recursive" if recursive else ""
            lines.append(
                f"spectraqc batch --folder \"{folder_path}\" --profile \"{profile_path}\" --mode {mode}{rec}"
            )
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _exit_code_for_status(status: Status) -> int:
    """Map Status enum to exit code."""
    if status == Status.PASS:
        return EXIT_PASS
    if status == Status.WARN:
        return EXIT_WARN
    return EXIT_FAIL


def _build_engine_meta() -> dict:
    """Build engine metadata for QCReport."""
    ffmpeg_version = "unknown"
    try:
        import subprocess
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            proc = subprocess.run(
                [ffmpeg, "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            if proc.stdout:
                ffmpeg_version = proc.stdout.splitlines()[0].strip()
    except Exception:
        ffmpeg_version = "unknown"
    return {
        "name": "spectraqc",
        "version": __version__,
        "build": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "deps": [
                {"name": "numpy", "version": np.__version__, "hash_sha256": "0" * 64},
                {"name": "ffmpeg", "version": ffmpeg_version, "hash_sha256": "0" * 64},
            ]
        }
    }


def _build_input_meta(audio_path: str, audio, fs: float, duration: float) -> dict:
    """Build input metadata for QCReport."""
    try:
        file_hash = sha256_hex_file(audio_path)
    except Exception:
        file_hash = "0" * 64
    
    # PCM hash from samples
    pcm_bytes = audio.samples.tobytes()
    import hashlib
    pcm_hash = hashlib.sha256(pcm_bytes).hexdigest()
    
    return {
        "path": str(Path(audio_path).resolve()),
        "file_hash_sha256": file_hash,
        "decoded_pcm_hash_sha256": pcm_hash,
        "fs_hz": fs,
        "channels": int(audio.channels),
        "duration_s": duration,
        "decode_backend": audio.backend,
        "decode_warnings": list(audio.warnings),
    }


def _analyze_audio(
    audio_path: str,
    profile_path: str,
    mode: str = "compliance",
    *,
    repair: dict | None = None
):
    """
    Run full analysis pipeline on audio file.
    
    Returns tuple of (qcreport_dict, decision, profile).
    """
    # Load profile
    profile = load_reference_profile(profile_path)
    
    # Load audio
    audio = load_audio(audio_path)
    
    # Analysis parameters (from profile or defaults)
    analysis_lock = profile.analysis_lock or {}
    nfft = int(analysis_lock.get("fft_size", 4096))
    hop = int(analysis_lock.get("hop_size", max(1, nfft // 2)))
    window = analysis_lock.get("window", "hann")
    psd_estimator = analysis_lock.get("psd_estimator", "welch")
    target_fs = analysis_lock.get("resample_fs_hz")
    target_fs = float(target_fs) if target_fs is not None else None
    dynamic_range_cfg = analysis_lock.get("dynamic_range", {})
    
    smoothing_cfg = profile.thresholds.get("_smoothing", {"type": "none"})
    ref_var = profile.thresholds.get("_ref_var_db2", np.ones_like(profile.freqs_hz))
    ref_tilt = spectral_tilt_db_per_oct(profile.freqs_hz, profile.ref_mean_db)
    channel_policy = str(analysis_lock.get("channel_policy", "mono")).strip().lower()
    silence_defaults = profile.thresholds.get("silence_detection", {})
    silence_override = analysis_lock.get("silence_detection", {})
    silence_gap_defaults = silence_defaults.get("gaps", {})
    silence_gap_override = silence_override.get("gaps", {})
    silence_cfg = {
        "min_rms_dbfs": float(
            silence_override.get(
                "min_rms_dbfs", silence_defaults.get("min_rms_dbfs", SILENCE_MIN_RMS_DBFS)
            )
        ),
        "frame_seconds": float(
            silence_override.get(
                "frame_seconds",
                silence_defaults.get("frame_seconds", SILENCE_FRAME_SECONDS),
            )
        ),
        "hop_seconds": float(
            silence_override.get(
                "hop_seconds", silence_defaults.get("hop_seconds", SILENCE_FRAME_SECONDS)
            )
        ),
        "min_duration_seconds": float(
            silence_override.get(
                "min_duration_seconds", silence_defaults.get("min_duration_seconds", 0.3)
            )
        ),
        "leading_threshold_seconds": float(
            silence_override.get(
                "leading_threshold_seconds",
                silence_defaults.get(
                    "leading_threshold_seconds", SILENCE_LEADING_THRESHOLD_SECONDS
                ),
            )
        ),
        "trailing_threshold_seconds": float(
            silence_override.get(
                "trailing_threshold_seconds",
                silence_defaults.get(
                    "trailing_threshold_seconds", SILENCE_TRAILING_THRESHOLD_SECONDS
                ),
            )
        ),
        "min_content_seconds": float(
            silence_override.get(
                "min_content_seconds",
                silence_defaults.get("min_content_seconds", SILENCE_MIN_CONTENT_SECONDS),
            )
        ),
        "gaps": {
            "warn_count": int(
                silence_gap_override.get(
                    "warn_count", silence_gap_defaults.get("warn_count", 1)
                )
            ),
            "fail_count": int(
                silence_gap_override.get(
                    "fail_count", silence_gap_defaults.get("fail_count", 3)
                )
            ),
            "warn_total_seconds": float(
                silence_gap_override.get(
                    "warn_total_seconds", silence_gap_defaults.get("warn_total_seconds", 0.2)
                )
            ),
            "fail_total_seconds": float(
                silence_gap_override.get(
                    "fail_total_seconds", silence_gap_defaults.get("fail_total_seconds", 0.5)
                )
            ),
            "max_gap_seconds": float(
                silence_gap_override.get(
                    "max_gap_seconds", silence_gap_defaults.get("max_gap_seconds", 0.0)
                )
            ),
        },
    }
    transient_defaults = profile.thresholds.get("transient_spikes", {})
    transient_override = analysis_lock.get("transient_spikes", {})
    transient_cfg = build_transient_spike_config(transient_defaults)
    if transient_override:
        transient_cfg = build_transient_spike_config({**transient_cfg, **transient_override})
    if channel_policy == "per_channel" and mode != "exploratory":
        raise ValueError("per_channel policy is only supported in exploratory mode.")
    algo_registry = build_algorithm_registry(
        analysis_lock=analysis_lock,
        smoothing_cfg=smoothing_cfg,
        channel_policy=channel_policy
    )
    algo_ids = algorithm_ids_from_registry(algo_registry)
    required_ids = {
        "ltpsd_welch_hann_powerhz_v1",
        "interp_linear_clamped_v1",
        "deviation_diff_db_v1",
        "band_metrics_df_weighted_v1",
        "spectral_tilt_regress_log2_v1",
        LOUDNESS_ALGO_ID,
        TRUE_PEAK_ALGO_ID,
        "channel_policy_v1",
        "dynamic_range_rms_percentile_v1",
        "dynamic_range_short_term_lufs_v1",
    }
    if smoothing_cfg.get("type") == "octave_fraction":
        required_ids.add("smoothing_octave_fraction_v1")
    elif smoothing_cfg.get("type") == "log_hz":
        required_ids.add("smoothing_log_hz_v1")
    else:
        required_ids.add("smoothing_none_v1")
    missing_ids = sorted(required_ids.difference(set(algo_ids)))
    if missing_ids:
        raise ValueError(f"Missing algorithm registry entries: {missing_ids}")
    analysis_buffers = apply_channel_policy(audio, channel_policy)
    level_anomaly_cfg = profile.thresholds.get("level_anomalies", {})
    level_anomalies = detect_level_anomalies(
        audio.samples,
        audio.fs,
        config=level_anomaly_cfg
    )
    level_flags = evaluate_level_anomalies(
        level_anomalies,
        config=level_anomaly_cfg
    )
    transient_spikes = detect_transient_spikes(
        audio.samples,
        audio.fs,
        config=transient_cfg,
    )

    def _analyze_single(mono_audio):
        resampled = False
        resampled_fs = mono_audio.fs
        resampled_samples = mono_audio.samples
        if target_fs is not None and target_fs > 0 and target_fs != mono_audio.fs:
            resampled_samples = _resample_linear(mono_audio.samples, mono_audio.fs, target_fs)
            resampled_fs = target_fs
            resampled = True
        analysis_buffer = mono_audio if not resampled else mono_audio.__class__(
            samples=resampled_samples,
            fs=resampled_fs,
            duration=resampled_samples.size / resampled_fs if resampled_fs > 0 else 0.0,
            channels=mono_audio.channels,
            backend=mono_audio.backend,
            warnings=list(mono_audio.warnings)
        )

        silence_metrics = detect_silence_segments(
            analysis_buffer.samples,
            analysis_buffer.fs,
            min_rms_dbfs=silence_cfg["min_rms_dbfs"],
            frame_seconds=silence_cfg["frame_seconds"],
            hop_seconds=silence_cfg["hop_seconds"],
            min_duration_seconds=silence_cfg["min_duration_seconds"],
            leading_threshold_seconds=silence_cfg["leading_threshold_seconds"],
            trailing_threshold_seconds=silence_cfg["trailing_threshold_seconds"],
            min_content_seconds=silence_cfg["min_content_seconds"],
        )
        silence_ratio = float(silence_metrics["silence_ratio"])
        effective_duration = float(
            silence_metrics.get("content_duration_s", analysis_buffer.duration)
        )

        ltpsd = compute_ltpsd(analysis_buffer, nfft=nfft, hop=hop)
        input_mean_db = interp_to_grid(ltpsd.freqs, ltpsd.mean_db, profile.freqs_hz)
        input_var_db2 = interp_var_ratio(ltpsd.freqs, ltpsd.var_db2, profile.freqs_hz)
        input_mean_db_raw = np.array(input_mean_db, dtype=np.float64)

        if smoothing_cfg.get("type") == "octave_fraction":
            oct_frac = smoothing_cfg.get("octave_fraction", 1/6)
            input_mean_db = smooth_octave_fraction(profile.freqs_hz, input_mean_db, oct_frac)
        elif smoothing_cfg.get("type") == "log_hz":
            bins_per_oct = smoothing_cfg.get("log_hz_bins_per_octave", 12)
            input_mean_db = smooth_log_hz(
                profile.freqs_hz,
                input_mean_db,
                bins_per_octave=int(bins_per_oct)
            )

        delta_db = deviation_curve_db(input_mean_db, profile.ref_mean_db)

        bm = band_metrics(
            profile.freqs_hz, delta_db, input_var_db2, ref_var, profile.bands
        )

        input_tilt = spectral_tilt_db_per_oct(profile.freqs_hz, input_mean_db)
        tilt_dev = input_tilt - ref_tilt

        tp_dbtp = true_peak_dbtp_mono(analysis_buffer.samples, analysis_buffer.fs)
        peak_dbfs = peak_dbfs_mono(analysis_buffer.samples)
        rms_dbfs = rms_dbfs_mono(analysis_buffer.samples)
        crest_factor_db = crest_factor_db_mono(analysis_buffer.samples)
        noise_floor_dbfs = noise_floor_dbfs_mono(
            analysis_buffer.samples,
            analysis_buffer.fs,
        )

        try:
            lufs_i, lra_lu = loudness_metrics_mono(
                analysis_buffer.samples, analysis_buffer.fs
            )
        except Exception:
            lufs_i = None
            lra_lu = None

        rms_cfg = dynamic_range_cfg.get("rms_percentile", {})
        dr_db = dynamic_range_percentile_dbfs_mono(
            analysis_buffer.samples,
            analysis_buffer.fs,
            frame_seconds=float(rms_cfg.get("frame_seconds", 3.0)),
            hop_seconds=float(rms_cfg.get("hop_seconds", 1.0)),
            low_percentile=float(rms_cfg.get("low_percentile", 10.0)),
            high_percentile=float(rms_cfg.get("high_percentile", 95.0)),
        )
        short_term_cfg = dynamic_range_cfg.get("short_term_lufs", {})
        try:
            dr_lu = dynamic_range_short_term_lufs_mono(
                analysis_buffer.samples,
                analysis_buffer.fs,
                low_percentile=float(short_term_cfg.get("low_percentile", 10.0)),
                high_percentile=float(short_term_cfg.get("high_percentile", 95.0)),
            )
        except Exception:
            dr_lu = None

        tonal_peaks = detect_tonal_peaks(
            profile.freqs_hz,
            input_mean_db_raw,
            bands=profile.bands,
            noise_floor_by_band=profile.noise_floor_by_band,
        )
        tonal_peak_max_delta = None
        tonal_deltas = [p.get("delta_db") for p in tonal_peaks if p.get("delta_db") is not None]
        if tonal_deltas:
            tonal_peak_max_delta = float(np.max(tonal_deltas))

        global_metrics = GlobalMetrics(
            spectral_tilt_db_per_oct=input_tilt,
            tilt_deviation_db_per_oct=tilt_dev,
            true_peak_dbtp=tp_dbtp,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            crest_factor_db=crest_factor_db,
            lufs_i=lufs_i,
            lra_lu=lra_lu,
            tonal_peak_max_delta_db=tonal_peak_max_delta,
            noise_floor_dbfs=noise_floor_dbfs,
            dynamic_range_db=dr_db,
            dynamic_range_lu=dr_lu,
        )

        spectral_artifacts = detect_spectral_artifacts(
            profile.freqs_hz,
            input_mean_db,
            expected_max_hz=float(profile.freqs_hz[-1]),
            config=profile.thresholds.get("spectral_artifacts", {})
        )
        spectral_flags = evaluate_spectral_artifacts(
            spectral_artifacts,
            expected_max_hz=float(profile.freqs_hz[-1]),
            config=profile.thresholds.get("spectral_artifacts", {})
        )

        decision = evaluate(bm, global_metrics, profile.thresholds)
        return {
            "analysis_buffer": analysis_buffer,
            "resampled": resampled,
            "silence_ratio": silence_ratio,
            "silence_metrics": silence_metrics,
            "effective_duration": effective_duration,
            "ltpsd": ltpsd,
            "input_mean_db": input_mean_db,
            "input_var_db2": input_var_db2,
            "delta_db": delta_db,
            "band_metrics": bm,
            "global_metrics": global_metrics,
            "noise_floor_dbfs": noise_floor_dbfs,
            "tonal_peaks": tonal_peaks,
            "decision": decision,
            "spectral_artifacts": spectral_artifacts,
            "spectral_flags": spectral_flags
        }

    results = [_analyze_single(buf) for buf in analysis_buffers]
    per_channel_noise_floor_dbfs = [r["noise_floor_dbfs"] for r in results]
    worst_idx = 0
    if len(results) > 1:
        order = {Status.PASS: 0, Status.WARN: 1, Status.FAIL: 2}
        worst_idx = max(
            range(len(results)),
            key=lambda i: order[results[i]["decision"].overall_status]
        )

    chosen = results[worst_idx]
    ltpsd = chosen["ltpsd"]
    input_mean_db = chosen["input_mean_db"]
    input_var_db2 = chosen["input_var_db2"]
    delta_db = chosen["delta_db"]
    bm = chosen["band_metrics"]
    global_metrics = chosen["global_metrics"]
    noise_floor_dbfs = chosen["noise_floor_dbfs"]
    tonal_peaks = chosen.get("tonal_peaks", [])
    decision = chosen["decision"]
    spectral_artifacts = chosen.get("spectral_artifacts", {})
    spectral_flags = chosen.get("spectral_flags", [])
    tp_dbtp = global_metrics.true_peak_dbtp
    lufs_i = global_metrics.lufs_i
    lra_lu = global_metrics.lra_lu
    analysis_buffer = chosen["analysis_buffer"]
    silence_ratio = chosen["silence_ratio"]
    silence_metrics = chosen["silence_metrics"]
    effective_duration = chosen["effective_duration"]
    resampled = chosen["resampled"]
    gap_summary = summarize_silence_gaps(silence_metrics, config=silence_cfg)
    if silence_metrics:
        silence_metrics["gaps"] = gap_summary
    gap_flags = evaluate_silence_gaps(silence_metrics, config=silence_cfg)
    
    # Build QCReport
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_id = f"qc_{uuid4().hex[:12]}"
    
    # Profile metadata for report
    profile_meta = {
        "name": profile.name,
        "kind": profile.kind,
        "version": profile.version,
        "profile_hash_sha256": profile.profile_hash_sha256,
        "signed": False,
        "signature": {"algo": "none", "value_b64": ""},
        "analysis_lock_hash": profile.analysis_lock_hash,
        "algorithm_ids": algo_ids
    }
    
    # Analysis configuration for report
    normalization_cfg = profile.normalization or {}
    loud_cfg = normalization_cfg.get("loudness", {})
    tp_cfg = normalization_cfg.get("true_peak", {})
    analysis_cfg = {
        "report_id": report_id,
        "created_utc": now_utc,
        "mode": mode,
        "resampled_fs_hz": analysis_buffer.fs,
        "channel_policy": str(channel_policy),
        "fft_size": nfft,
        "hop_size": hop,
        "window": str(window),
        "psd_estimator": str(psd_estimator),
        "smoothing": smoothing_cfg,
        "algorithm_registry": algo_registry,
        "bands": [
            {"name": b.name, "f_low_hz": b.f_low, "f_high_hz": b.f_high}
            for b in profile.bands
        ],
        "dynamic_range": dynamic_range_cfg,
        "normalization": {
            "loudness": {
                "enabled": bool(loud_cfg.get("enabled", False)),
                "target_lufs_i": float(loud_cfg.get("target_lufs_i", -14.0)),
                "measured_lufs_i": lufs_i if lufs_i is not None else -100.0,
                "applied_gain_db": 0.0,
                "algorithm_id": LOUDNESS_ALGO_ID
            },
            "true_peak": {
                "enabled": bool(tp_cfg.get("enabled", False)),
                "max_dbtp": float(tp_cfg.get("max_dbtp", -1.0)),
                "measured_dbtp": tp_dbtp,
                "algorithm_id": TRUE_PEAK_ALGO_ID
            }
        },
        "silence_gate": {
            "enabled": False,
            "min_rms_dbfs": silence_cfg["min_rms_dbfs"],
            "frame_seconds": silence_cfg["frame_seconds"],
            "hop_seconds": silence_cfg["hop_seconds"],
            "min_duration_seconds": silence_cfg["min_duration_seconds"],
            "leading_threshold_seconds": silence_cfg["leading_threshold_seconds"],
            "trailing_threshold_seconds": silence_cfg["trailing_threshold_seconds"],
            "min_content_seconds": silence_cfg["min_content_seconds"],
            "max_gap_seconds": silence_cfg["gaps"]["max_gap_seconds"],
            "silence_ratio": silence_ratio,
            "effective_seconds": effective_duration
        },
        "transient_spikes": transient_cfg,
    }
    
    # Band metrics for report
    band_metrics_list = [
        {
            "band_name": m.band.name,
            "f_low_hz": m.band.f_low,
            "f_high_hz": m.band.f_high,
            "mean_deviation_db": m.mean_deviation_db,
            "max_deviation_db": m.max_deviation_db,
            "variance_ratio": m.variance_ratio
        }
        for m in bm
    ]
    
    # Global metrics for report
    global_metrics_dict = {
        "spectral_tilt_db_per_oct": global_metrics.spectral_tilt_db_per_oct,
        "tilt_deviation_db_per_oct": global_metrics.tilt_deviation_db_per_oct,
    }
    if global_metrics.true_peak_dbtp is not None:
        global_metrics_dict["true_peak_dbtp"] = global_metrics.true_peak_dbtp
    if global_metrics.peak_dbfs is not None:
        global_metrics_dict["peak_dbfs"] = global_metrics.peak_dbfs
    if global_metrics.rms_dbfs is not None:
        global_metrics_dict["rms_dbfs"] = global_metrics.rms_dbfs
    if global_metrics.lufs_i is not None:
        global_metrics_dict["lufs_i"] = global_metrics.lufs_i
    if global_metrics.noise_floor_dbfs is not None:
        global_metrics_dict["noise_floor_dbfs"] = global_metrics.noise_floor_dbfs
    if global_metrics.crest_factor_db is not None:
        global_metrics_dict["crest_factor_db"] = global_metrics.crest_factor_db
    if global_metrics.lra_lu is not None:
        global_metrics_dict["lra_lu"] = global_metrics.lra_lu
    if global_metrics.dynamic_range_db is not None:
        global_metrics_dict["dynamic_range_db"] = global_metrics.dynamic_range_db
    if global_metrics.dynamic_range_lu is not None:
        global_metrics_dict["dynamic_range_lu"] = global_metrics.dynamic_range_lu
    if global_metrics.tonal_peak_max_delta_db is not None:
        global_metrics_dict["tonal_peak_max_delta_db"] = global_metrics.tonal_peak_max_delta_db
    if tonal_peaks:
        global_metrics_dict["tonal_peaks"] = tonal_peaks
    if spectral_artifacts:
        global_metrics_dict["spectral_artifacts"] = spectral_artifacts
    if level_anomalies:
        global_metrics_dict["level_anomalies"] = level_anomalies
    if silence_metrics:
        global_metrics_dict["silence"] = silence_metrics
    if transient_spikes:
        global_metrics_dict["transient_spikes"] = transient_spikes
    
    # Decisions for report
    decisions_dict = {
        "overall_status": decision.overall_status.value,
        "technical_status": decision.overall_status.value,
        "band_decisions": [
            {
                "band_name": bd.band.name,
                "mean": {
                    "metric": bd.mean.metric,
                    "value": bd.mean.value,
                    "units": bd.mean.units,
                    "status": bd.mean.status.value,
                    "pass_limit": bd.mean.pass_limit,
                    "warn_limit": bd.mean.warn_limit,
                    "notes": bd.mean.notes
                },
                "max": {
                    "metric": bd.max.metric,
                    "value": bd.max.value,
                    "units": bd.max.units,
                    "status": bd.max.status.value,
                    "pass_limit": bd.max.pass_limit,
                    "warn_limit": bd.max.warn_limit,
                    "notes": bd.max.notes
                },
                "variance": {
                    "metric": bd.variance.metric,
                    "value": bd.variance.value,
                    "units": bd.variance.units,
                    "status": bd.variance.status.value,
                    "pass_limit": bd.variance.pass_limit,
                    "warn_limit": bd.variance.warn_limit,
                    "notes": bd.variance.notes
                }
            }
            for bd in decision.band_decisions
        ],
        "global_decisions": [
            {
                "metric": gd.metric,
                "value": gd.value,
                "units": gd.units,
                "status": gd.status.value,
                "pass_limit": gd.pass_limit,
                "warn_limit": gd.warn_limit,
                "notes": gd.notes
            }
            for gd in decision.global_decisions
        ],
        "flags": spectral_flags + level_flags + gap_flags
    }
    
    # Confidence assessment
    confidence = _build_confidence(
        audio,
        effective_duration=effective_duration,
        silence_ratio=silence_ratio,
        resampled=resampled
    )
    
    # Build final report
    qcreport = build_qcreport_dict(
        engine=_build_engine_meta(),
        input_meta=_build_input_meta(audio_path, audio, audio.fs, audio.duration),
        profile=profile_meta,
        analysis=analysis_cfg,
        freqs_hz=profile.freqs_hz,
        ref_mean_db=profile.ref_mean_db,
        ref_var_db2=profile.thresholds.get("_ref_var_db2", np.ones_like(profile.freqs_hz)),
        ltpsd_mean_db=input_mean_db,
        ltpsd_var_db2=input_var_db2,
        delta_mean_db=delta_db,
        band_metrics=band_metrics_list,
        global_metrics=global_metrics_dict,
        noise_floor={
            "policy": str(channel_policy),
            "by_channel_dbfs": per_channel_noise_floor_dbfs,
            "merged_dbfs": noise_floor_dbfs,
        },
        decisions=decisions_dict,
        confidence=confidence,
        repair=repair
    )
    
    return qcreport, decision, profile, algo_ids


def cmd_repair(args) -> int:
    """Handle repair command."""
    try:
        profile = load_reference_profile(args.profile)
        plan = _load_repair_plan(args.repair_plan)
        audio = load_audio(args.audio_path)
        repaired_samples, steps = apply_repair_plan(audio.samples, audio.fs, plan)
        out_path = Path(args.out) if args.out else Path(args.audio_path).with_suffix(".repaired.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, repaired_samples, int(audio.fs))

        before_metrics = compute_repair_metrics(audio.samples, audio.fs, profile)
        after_metrics = compute_repair_metrics(repaired_samples, audio.fs, profile)
        delta_curve = (
            np.array(after_metrics["deviation_curve_db"], dtype=np.float64)
            - np.array(before_metrics["deviation_curve_db"], dtype=np.float64)
        ).tolist()
        delta_metrics = {
            "true_peak_dbtp": after_metrics["true_peak_dbtp"] - before_metrics["true_peak_dbtp"],
            "noise_floor_dbfs": after_metrics["noise_floor_dbfs"] - before_metrics["noise_floor_dbfs"],
            "deviation_curve_db": delta_curve
        }
        repair_section = {
            "plan_source": {"path": str(args.repair_plan)},
            "steps": steps,
            "metrics": {
                "before": before_metrics,
                "after": after_metrics,
                "delta": delta_metrics
            },
            "output": {
                "path": str(out_path),
                "file_hash_sha256": sha256_hex_file(str(out_path)),
                "channels": audio.channels,
                "sample_rate_hz": audio.fs
            }
        }

        qcreport, decision, _, _ = _analyze_audio(
            str(out_path),
            args.profile,
            mode="compliance",
            repair=repair_section
        )

        output_json = json.dumps(qcreport, indent=2)
        if args.report:
            Path(args.report).write_text(output_json, encoding="utf-8")
            print(f"Report written to: {args.report}", file=sys.stderr)
        else:
            print(output_json)
        print(f"Repaired audio written to: {out_path}", file=sys.stderr)
        return _exit_code_for_status(decision.overall_status)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except json.JSONDecodeError as e:
        print(f"Error: Invalid profile JSON - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except KeyError as e:
        print(f"Error: Missing profile key - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_analyze(args) -> int:
    """Handle analyze command."""
    try:
        qcreport, decision, profile, algo_ids = _analyze_audio(
            args.audio_path,
            args.profile,
            mode=args.mode
        )
        
        # Output report
        output_json = json.dumps(qcreport, indent=2)
        if args.out:
            Path(args.out).write_text(output_json, encoding="utf-8")
            print(f"Report written to: {args.out}", file=sys.stderr)
        else:
            print(output_json)

        if args.repro_md:
            Path(args.repro_md).write_text(
                _render_repro_doc(
                    profile=profile,
                    algorithm_ids=algo_ids,
                    profile_path=args.profile,
                    mode=args.mode,
                    audio_path=args.audio_path
                ),
                encoding="utf-8"
            )
        
        return _exit_code_for_status(decision.overall_status)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except json.JSONDecodeError as e:
        print(f"Error: Invalid profile JSON - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except KeyError as e:
        print(f"Error: Missing profile key - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_validate(args) -> int:
    """Handle validate command."""
    try:
        _, decision, profile, _ = _analyze_audio(
            args.audio_path,
            args.profile,
            mode="compliance"
        )
        
        status = decision.overall_status
        
        # Print summary
        print(f"Profile: {profile.name} ({profile.kind})")
        print(f"Status: {status.value.upper()}")
        
        # Print band summaries
        for bd in decision.band_decisions:
            worst = max(
                bd.mean.status, bd.max.status, bd.variance.status,
                key=lambda s: {"pass": 0, "warn": 1, "fail": 2}[s.value]
            )
            if worst != Status.PASS:
                print(f"  {bd.band.name}: {worst.value} (mean={bd.mean.value:+.2f}dB, max={bd.max.value:.2f}dB)")
        
        # Print global summaries
        for gd in decision.global_decisions:
            if gd.status != Status.PASS:
                print(f"  {gd.metric}: {gd.status.value} ({gd.value:.3f} {gd.units})")
        
        # Determine exit based on fail-on mode
        if args.fail_on == "warn" and status in (Status.WARN, Status.FAIL):
            return _exit_code_for_status(status)
        elif args.fail_on == "fail" and status == Status.FAIL:
            return EXIT_FAIL
        elif status == Status.FAIL:
            return EXIT_FAIL
        
        return _exit_code_for_status(status)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_DECODE_ERROR
    except json.JSONDecodeError as e:
        print(f"Error: Invalid profile JSON - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except KeyError as e:
        print(f"Error: Missing profile key - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_inspect_ref(args) -> int:
    """Handle inspect-ref command."""
    try:
        profile = load_reference_profile(args.profile)
        
        print(f"Profile: {profile.name}")
        print(f"Kind: {profile.kind}")
        print(f"Version: {profile.version}")
        print(f"Hash: {profile.profile_hash_sha256[:16]}...")
        print()
        print("Frequency Bands:")
        for b in profile.bands:
            print(f"  {b.name}: {b.f_low:.0f} - {b.f_high:.0f} Hz")
        print()
        print("Thresholds:")
        print(f"  Band mean (default): pass={profile.thresholds['band_mean_db']['default'][0]:.1f}dB, warn={profile.thresholds['band_mean_db']['default'][1]:.1f}dB")
        print(f"  Band max (default): pass={profile.thresholds['band_max_db']['all'][0]:.1f}dB, warn={profile.thresholds['band_max_db']['all'][1]:.1f}dB")
        print(f"  Tilt: pass={profile.thresholds['tilt_db_per_oct'][0]:.2f}dB/oct, warn={profile.thresholds['tilt_db_per_oct'][1]:.2f}dB/oct")
        if "noise_floor_dbfs" in profile.thresholds:
            print(
                "  Noise floor: pass="
                f"{profile.thresholds['noise_floor_dbfs'][0]:.1f}dBFS, warn="
                f"{profile.thresholds['noise_floor_dbfs'][1]:.1f}dBFS"
            )
        print()
        print(f"Frequency grid: {len(profile.freqs_hz)} bins, {profile.freqs_hz[0]:.1f} - {profile.freqs_hz[-1]:.1f} Hz")
        
        return EXIT_PASS
        
    except FileNotFoundError as e:
        print(f"Error: Profile not found - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except json.JSONDecodeError as e:
        print(f"Error: Invalid profile JSON - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except KeyError as e:
        print(f"Error: Missing profile key - {e}", file=sys.stderr)
        return EXIT_PROFILE_ERROR
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_build_profile(args) -> int:
    """Handle build-profile command."""
    try:
        from spectraqc.profiles.builder import (
            build_reference_profile_from_folder,
            build_reference_profile_from_manifest,
        )

        if bool(args.manifest) == bool(args.folder):
            print("Error: provide exactly one of --manifest or --folder.", file=sys.stderr)
            return EXIT_BAD_ARGS

        if args.manifest:
            profile, output_path = build_reference_profile_from_manifest(
                args.manifest,
                profile_name=args.name,
                profile_kind=args.kind,
                output_path=args.out,
            )
        else:
            profile, output_path = build_reference_profile_from_folder(
                args.folder,
                recursive=args.recursive,
                profile_name=args.name,
                profile_kind=args.kind,
                output_path=args.out,
            )

        print(f"Profile written to: {output_path}")
        print(f"Profile name: {profile['profile']['name']}")
        print(f"Profile kind: {profile['profile']['kind']}")
        print(f"Profile version: {profile['profile']['version']}")
        print(f"Files used: {profile['corpus_stats']['file_count']}")
        return EXIT_PASS
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_batch(args) -> int:
    """Handle batch command."""
    try:
        audio_paths: list[Path] = []
        if args.folder:
            audio_paths.extend(_iter_audio_files(Path(args.folder), args.recursive))
        if args.manifest:
            from spectraqc.corpus.manifest import load_corpus_manifest
            _, entries, _ = load_corpus_manifest(args.manifest)
            for e in entries:
                if e.exclude:
                    continue
                audio_paths.append(Path(e.path))
        if not audio_paths:
            print("Error: No input files found.", file=sys.stderr)
            return EXIT_BAD_ARGS

        out_dir = Path(args.out_dir) if args.out_dir else None
        report_outputs_requested = any([
            args.summary_json,
            args.summary_md,
            args.summary_kpis_json,
            args.summary_kpis_csv,
            args.report_md,
            args.report_html,
            args.repro_md
        ])
        if report_outputs_requested and out_dir is None:
            print(
                "Error: --out-dir is required when writing batch reports so outputs live alongside QC reports.",
                file=sys.stderr
            )
            return EXIT_BAD_ARGS
        if report_outputs_requested and out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        max_workers = max(1, int(args.workers))
        max_workers = min(max_workers, len(audio_paths))

        failures = 0
        results: list[tuple[str, str, str | None, dict | None]] = []
        if max_workers == 1:
            for p in audio_paths:
                audio_path = str(p)
                result = _batch_worker((audio_path, args.profile, args.mode, str(out_dir) if out_dir else None))
                results.append(result)
                if result[2] is not None:
                    failures += 1
                    print(f"[ERROR] {result[0]}: {result[2]}", file=sys.stderr)
                else:
                    print(f"[OK] {result[0]}: {result[1]}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(_batch_worker, (str(p), args.profile, args.mode, str(out_dir) if out_dir else None))
                    for p in audio_paths
                ]
                for fut in as_completed(futures):
                    audio_path, status, err, report = fut.result()
                    results.append((audio_path, status, err, report))
                    if err:
                        failures += 1
                        print(f"[ERROR] {audio_path}: {err}", file=sys.stderr)
                    else:
                        print(f"[OK] {audio_path}: {status}")

        summary = build_batch_summary(results)
        summary_json_path = _resolve_report_output_path(
            out_dir,
            args.summary_json,
            "batch-summary.json"
        )
        if summary_json_path:
            summary_json_path.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8"
            )
        
        kpis_json_path = _resolve_report_output_path(
            out_dir,
            args.summary_kpis_json,
            "batch-kpis.json"
        )
        if kpis_json_path:
            kpis_payload = build_kpi_payload(summary)
            kpis_json_path.write_text(
                json.dumps(kpis_payload, indent=2),
                encoding="utf-8"
            )
        
        kpis_csv_path = _resolve_report_output_path(
            out_dir,
            args.summary_kpis_csv,
            "batch-kpis.csv"
        )
        if kpis_csv_path:
            kpis_csv_path.write_text(
                render_kpis_csv(summary),
                encoding="utf-8"
            )
        
        summary_md_path = _resolve_report_output_path(
            out_dir,
            args.summary_md,
            "batch-summary.md"
        )
        if summary_md_path:
            summary_md_path.write_text(
                _render_markdown_summary(summary),
                encoding="utf-8"
            )
        report_md_path = _resolve_report_output_path(
            out_dir,
            args.report_md,
            "batch-report.md"
        )
        if report_md_path:
            report_md_path.write_text(
                _render_corpus_report_md(
                    summary,
                    profile_path=args.profile,
                    mode=args.mode
                ),
                encoding="utf-8"
            )
        report_html_path = _resolve_report_output_path(
            out_dir,
            args.report_html,
            "batch-report.html"
        )
        if report_html_path:
            file_links = None
            embedded_reports = None
            if out_dir:
                file_links = []
                embedded_reports = {}
                for audio_path, status, err, _ in results:
                    label = Path(audio_path).name
                    link_path = _output_path(out_dir, Path(audio_path))
                    href = os.path.relpath(link_path, report_html_path.parent)
                    file_links.append((label, status, href))
                for audio_path, _, _, report in results:
                    if report:
                        label = Path(audio_path).name
                        embedded_reports[label] = report
            report_html_path.write_text(
                _render_corpus_report_html(
                    summary,
                    profile_path=args.profile,
                    mode=args.mode,
                    file_links=file_links,
                    embedded_reports=embedded_reports
                ),
                encoding="utf-8"
            )
        repro_md_path = _resolve_report_output_path(
            out_dir,
            args.repro_md,
            "batch-repro.md"
        )
        if repro_md_path:
            profile = load_reference_profile(args.profile)
            algo_registry = build_algorithm_registry(
                analysis_lock=profile.analysis_lock or {},
                smoothing_cfg=profile.thresholds.get("_smoothing", {"type": "none"}),
                channel_policy=str(profile.analysis_lock.get("channel_policy", "mono"))
            )
            algo_ids = algorithm_ids_from_registry(algo_registry)
            repro_md_path.write_text(
                _render_repro_doc(
                    profile=profile,
                    algorithm_ids=algo_ids,
                    profile_path=args.profile,
                    mode=args.mode,
                    manifest_path=args.manifest,
                    folder_path=args.folder,
                    recursive=args.recursive
                ),
                encoding="utf-8"
            )

        if failures:
            return EXIT_INTERNAL_ERROR
        return EXIT_PASS

    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cmd_gui(args) -> int:
    """Launch the SpectraQC GUI."""
    from spectraqc.gui.server import run_gui_server

    run_gui_server(host=args.host, port=args.port, open_browser=not args.no_open)
    return EXIT_PASS


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spectraqc",
        description="SpectraQC - Spectral Quality Control Tool"
    )
    parser.add_argument(
        "--version", action="version",
        version=f"spectraqc {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze audio file against reference profile"
    )
    analyze_parser.add_argument(
        "audio_path",
        help="Path to audio file (WAV)"
    )
    analyze_parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to reference profile JSON"
    )
    analyze_parser.add_argument(
        "--mode", "-m",
        choices=["compliance", "exploratory"],
        default="compliance",
        help="Analysis mode (default: compliance)"
    )
    analyze_parser.add_argument(
        "--out", "-o",
        help="Output path for QC report JSON"
    )
    analyze_parser.add_argument(
        "--repro-md",
        help="Output path for reproducibility Markdown"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # repair command
    repair_parser = subparsers.add_parser(
        "repair",
        help="Repair audio with a DSP plan and produce a QC report"
    )
    repair_parser.add_argument(
        "audio_path",
        help="Path to audio file (WAV)"
    )
    repair_parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to reference profile JSON"
    )
    repair_parser.add_argument(
        "--repair-plan",
        required=True,
        help="Path to repair plan (YAML/JSON)"
    )
    repair_parser.add_argument(
        "--out",
        help="Output path for repaired audio (default: <input>.repaired.wav)"
    )
    repair_parser.add_argument(
        "--report",
        help="Output path for QC report JSON (default: stdout)"
    )
    repair_parser.set_defaults(func=cmd_repair)
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate audio file (simple pass/warn/fail output)"
    )
    validate_parser.add_argument(
        "audio_path",
        help="Path to audio file (WAV)"
    )
    validate_parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to reference profile JSON"
    )
    validate_parser.add_argument(
        "--fail-on",
        choices=["fail", "warn"],
        default="fail",
        help="When to return non-zero exit code (default: fail)"
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # inspect-ref command
    inspect_parser = subparsers.add_parser(
        "inspect-ref",
        help="Inspect reference profile"
    )
    inspect_parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to reference profile JSON"
    )
    inspect_parser.set_defaults(func=cmd_inspect_ref)

    # build-profile command
    build_parser = subparsers.add_parser(
        "build-profile",
        help="Build a reference profile from a manifest or folder"
    )
    build_parser.add_argument(
        "--manifest",
        help="Corpus manifest JSON path"
    )
    build_parser.add_argument(
        "--folder",
        help="Folder containing reference audio"
    )
    build_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when using --folder"
    )
    build_parser.add_argument(
        "--out",
        help="Output path for the profile JSON"
    )
    build_parser.add_argument(
        "--name",
        default="streaming_generic_v1",
        help="Profile name"
    )
    build_parser.add_argument(
        "--kind",
        default="streaming",
        help="Profile kind (broadcast, streaming, archive, custom)"
    )
    build_parser.set_defaults(func=cmd_build_profile)

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch analyze a folder or manifest"
    )
    batch_parser.add_argument(
        "--folder",
        help="Folder containing audio files"
    )
    batch_parser.add_argument(
        "--manifest",
        help="Corpus manifest JSON path"
    )
    batch_parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to reference profile JSON"
    )
    batch_parser.add_argument(
        "--mode", "-m",
        choices=["compliance", "exploratory"],
        default="compliance",
        help="Analysis mode (default: compliance)"
    )
    batch_parser.add_argument(
        "--out-dir",
        help="Output directory for QC report JSONs"
    )
    batch_parser.add_argument(
        "--summary-json",
        default="batch-summary.json",
        help="Batch summary JSON filename (default: batch-summary.json in --out-dir)"
    )
    batch_parser.add_argument(
        "--summary-md",
        default="batch-summary.md",
        help="Batch summary Markdown filename (default: batch-summary.md in --out-dir)"
    )
    batch_parser.add_argument(
        "--summary-kpis-json",
        default="batch-kpis.json",
        help="Batch KPIs JSON filename (default: batch-kpis.json in --out-dir)"
    )
    batch_parser.add_argument(
        "--summary-kpis-csv",
        default="batch-kpis.csv",
        help="Batch KPIs CSV filename (default: batch-kpis.csv in --out-dir)"
    )
    batch_parser.add_argument(
        "--report-md",
        default="batch-report.md",
        help="One-page batch report Markdown filename (default: batch-report.md in --out-dir)"
    )
    batch_parser.add_argument(
        "--report-html",
        default="batch-report.html",
        help="One-page batch report HTML filename (default: batch-report.html in --out-dir)"
    )
    batch_parser.add_argument(
        "--repro-md",
        default="batch-repro.md",
        help="Reproducibility Markdown filename (default: batch-repro.md in --out-dir)"
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when using --folder"
    )
    batch_parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers (default: cpu_count-1)"
    )
    batch_parser.set_defaults(func=cmd_batch)

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the SpectraQC GUI"
    )
    gui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)"
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve the GUI on (default: 8000)"
    )
    gui_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically"
    )
    gui_parser.set_defaults(func=cmd_gui)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(EXIT_BAD_ARGS)


if __name__ == "__main__":
    main()
