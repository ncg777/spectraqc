"""SpectraQC CLI - Spectral Quality Control Tool."""
from __future__ import annotations
import argparse
import json
import sys
import platform
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from spectraqc.version import __version__
from spectraqc.types import Status, GlobalMetrics
from spectraqc.io.audio import load_audio, apply_channel_policy
from spectraqc.analysis.ltpsd import compute_ltpsd
from spectraqc.metrics.grid import interp_to_grid, interp_var_ratio
from spectraqc.metrics.smoothing import smooth_octave_fraction
from spectraqc.metrics.deviation import deviation_curve_db
from spectraqc.metrics.integration import band_metrics
from spectraqc.metrics.tilt import spectral_tilt_db_per_oct
from spectraqc.metrics.truepeak import true_peak_dbtp_mono
from spectraqc.metrics.loudness import integrated_lufs_mono
from spectraqc.algorithms.registry import (
    build_algorithm_registry,
    algorithm_ids_from_registry,
    LOUDNESS_ALGO_ID,
    TRUE_PEAK_ALGO_ID,
)
from spectraqc.profiles.loader import load_reference_profile
from spectraqc.thresholds.evaluator import evaluate
from spectraqc.reporting.qcreport import build_qcreport_dict
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
SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}


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


def _aggregate_batch_results(results: list[tuple[str, str, str | None, dict | None]]) -> dict:
    """Aggregate batch results into summary statistics."""
    counts = {"pass": 0, "warn": 0, "fail": 0, "error": 0}
    failure_causes: dict[str, int] = {}
    confidence = {"pass": 0, "warn": 0, "fail": 0}
    confidence_reasons: dict[str, int] = {}
    metrics = {
        "tilt_deviation": [],
        "true_peak": [],
        "band_mean_deviation": [],
        "band_max_deviation": [],
        "variance_ratio": [],
    }

    for _, status, err, report in results:
        if status not in counts:
            counts["error"] += 1
        else:
            counts[status] += 1
        if err:
            failure_causes[err] = failure_causes.get(err, 0) + 1
            continue
        if not report:
            continue
        conf_status = report.get("confidence", {}).get("status", "pass")
        if conf_status in confidence:
            confidence[conf_status] += 1
        for reason in report.get("confidence", {}).get("reasons", []):
            confidence_reasons[reason] = confidence_reasons.get(reason, 0) + 1
        # Collect decision notes as failure causes for warn/fail
        decisions = report.get("decisions", {})
        for bd in decisions.get("band_decisions", []):
            for key in ("mean", "max", "variance"):
                d = bd.get(key, {})
                d_status = d.get("status")
                if d_status in ("warn", "fail"):
                    note = d.get("notes") or d.get("metric", "unknown")
                    failure_causes[note] = failure_causes.get(note, 0) + 1
                    if d.get("metric") == "band_mean_deviation":
                        metrics["band_mean_deviation"].append(d.get("value"))
                    if d.get("metric") == "band_max_deviation":
                        metrics["band_max_deviation"].append(d.get("value"))
                    if d.get("metric") == "variance_ratio":
                        metrics["variance_ratio"].append(d.get("value"))
        for gd in decisions.get("global_decisions", []):
            gd_status = gd.get("status")
            if gd_status in ("warn", "fail"):
                note = gd.get("notes") or gd.get("metric", "unknown")
                failure_causes[note] = failure_causes.get(note, 0) + 1
                if gd.get("metric") == "tilt_deviation":
                    metrics["tilt_deviation"].append(gd.get("value"))
                if gd.get("metric") == "true_peak":
                    metrics["true_peak"].append(gd.get("value"))

    def _summary(values: list) -> dict | None:
        vals = [v for v in values if isinstance(v, (int, float))]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr))
        }

    distributions = {k: _summary(v) for k, v in metrics.items()}
    distributions = {k: v for k, v in distributions.items() if v is not None}

    return {
        "counts": counts,
        "confidence_counts": confidence,
        "confidence_reasons": dict(sorted(confidence_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        "failure_causes": dict(sorted(failure_causes.items(), key=lambda kv: kv[1], reverse=True)),
        "distributions": distributions
    }


def _render_markdown_summary(summary: dict) -> str:
    """Render a Markdown summary."""
    lines = []
    counts = summary.get("counts", {})
    lines.append("# Batch Summary")
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(f"- pass: {counts.get('pass', 0)}")
    lines.append(f"- warn: {counts.get('warn', 0)}")
    lines.append(f"- fail: {counts.get('fail', 0)}")
    lines.append(f"- error: {counts.get('error', 0)}")
    lines.append("")
    conf = summary.get("confidence_counts", {})
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
    lines.append("## Metric Distributions")
    lines.append("")
    for metric, stats in summary.get("distributions", {}).items():
        lines.append(f"- {metric}: count={stats['count']} min={stats['min']:.3f} p50={stats['p50']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}")
    lines.append("")
    return "\n".join(lines)


def _render_corpus_report_md(summary: dict, *, profile_path: str, mode: str) -> str:
    """Render a one-page Markdown report for batch results."""
    counts = summary.get("counts", {})
    conf = summary.get("confidence_counts", {})
    causes = list(summary.get("failure_causes", {}).items())
    conf_reasons = list(summary.get("confidence_reasons", {}).items())
    dists = summary.get("distributions", {})

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
    lines.append("## Band Metrics (Distributions)")
    lines.append("")
    for metric in ("band_mean_deviation", "band_max_deviation", "variance_ratio"):
        stats = dists.get(metric)
        if not stats:
            continue
        lines.append(
            f"- {metric}: count={stats['count']} min={stats['min']:.3f} "
            f"p50={stats['p50']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}"
        )
    lines.append("")
    lines.append("## Global Metrics (Distributions)")
    lines.append("")
    for metric in ("tilt_deviation", "true_peak"):
        stats = dists.get(metric)
        if not stats:
            continue
        lines.append(
            f"- {metric}: count={stats['count']} min={stats['min']:.3f} "
            f"p50={stats['p50']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}"
        )
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


def _render_corpus_report_html(summary: dict, *, profile_path: str, mode: str) -> str:
    """Render a one-page HTML report for batch results."""
    md = _render_corpus_report_md(summary, profile_path=profile_path, mode=mode)
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = body.replace("\n", "<br>\n")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>SpectraQC Batch Report</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}</style>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


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


def _analyze_audio(audio_path: str, profile_path: str, mode: str = "compliance"):
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
    
    smoothing_cfg = profile.thresholds.get("_smoothing", {"type": "none"})
    ref_var = profile.thresholds.get("_ref_var_db2", np.ones_like(profile.freqs_hz))
    ref_tilt = spectral_tilt_db_per_oct(profile.freqs_hz, profile.ref_mean_db)
    channel_policy = str(analysis_lock.get("channel_policy", "mono")).strip().lower()
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
    }
    if smoothing_cfg.get("type") == "octave_fraction":
        required_ids.add("smoothing_octave_fraction_v1")
    else:
        required_ids.add("smoothing_none_v1")
    missing_ids = sorted(required_ids.difference(set(algo_ids)))
    if missing_ids:
        raise ValueError(f"Missing algorithm registry entries: {missing_ids}")
    analysis_buffers = apply_channel_policy(audio, channel_policy)

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

        silence_ratio = _compute_silence_ratio(
            analysis_buffer.samples,
            analysis_buffer.fs,
            min_rms_dbfs=SILENCE_MIN_RMS_DBFS,
            frame_seconds=SILENCE_FRAME_SECONDS
        )
        effective_duration = analysis_buffer.duration * (1.0 - silence_ratio)

        ltpsd = compute_ltpsd(analysis_buffer, nfft=nfft, hop=hop)
        input_mean_db = interp_to_grid(ltpsd.freqs, ltpsd.mean_db, profile.freqs_hz)
        input_var_db2 = interp_var_ratio(ltpsd.freqs, ltpsd.var_db2, profile.freqs_hz)

        if smoothing_cfg.get("type") == "octave_fraction":
            oct_frac = smoothing_cfg.get("octave_fraction", 1/6)
            input_mean_db = smooth_octave_fraction(profile.freqs_hz, input_mean_db, oct_frac)

        delta_db = deviation_curve_db(input_mean_db, profile.ref_mean_db)

        bm = band_metrics(
            profile.freqs_hz, delta_db, input_var_db2, ref_var, profile.bands
        )

        input_tilt = spectral_tilt_db_per_oct(profile.freqs_hz, input_mean_db)
        tilt_dev = input_tilt - ref_tilt

        tp_dbtp = true_peak_dbtp_mono(analysis_buffer.samples, analysis_buffer.fs)

        try:
            lufs_i = integrated_lufs_mono(analysis_buffer.samples, analysis_buffer.fs)
        except Exception:
            lufs_i = None

        global_metrics = GlobalMetrics(
            spectral_tilt_db_per_oct=input_tilt,
            tilt_deviation_db_per_oct=tilt_dev,
            true_peak_dbtp=tp_dbtp,
            lufs_i=lufs_i
        )

        decision = evaluate(bm, global_metrics, profile.thresholds)
        return {
            "analysis_buffer": analysis_buffer,
            "resampled": resampled,
            "silence_ratio": silence_ratio,
            "effective_duration": effective_duration,
            "ltpsd": ltpsd,
            "input_mean_db": input_mean_db,
            "input_var_db2": input_var_db2,
            "delta_db": delta_db,
            "band_metrics": bm,
            "global_metrics": global_metrics,
            "decision": decision
        }

    results = [_analyze_single(buf) for buf in analysis_buffers]
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
    decision = chosen["decision"]
    tp_dbtp = global_metrics.true_peak_dbtp
    lufs_i = global_metrics.lufs_i
    analysis_buffer = chosen["analysis_buffer"]
    silence_ratio = chosen["silence_ratio"]
    effective_duration = chosen["effective_duration"]
    resampled = chosen["resampled"]
    
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
            "min_rms_dbfs": SILENCE_MIN_RMS_DBFS,
            "silence_ratio": silence_ratio,
            "effective_seconds": effective_duration
        }
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
        ]
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
        ltpsd_mean_db=input_mean_db,
        ltpsd_var_db2=input_var_db2,
        delta_mean_db=delta_db,
        band_metrics=band_metrics_list,
        global_metrics=global_metrics_dict,
        decisions=decisions_dict,
        confidence=confidence
    )
    
    return qcreport, decision, profile, algo_ids


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

        summary = _aggregate_batch_results(results)
        if args.summary_json:
            Path(args.summary_json).write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8"
            )
        if args.summary_md:
            Path(args.summary_md).write_text(
                _render_markdown_summary(summary),
                encoding="utf-8"
            )
        if args.report_md:
            Path(args.report_md).write_text(
                _render_corpus_report_md(summary, profile_path=args.profile, mode=args.mode),
                encoding="utf-8"
            )
        if args.report_html:
            Path(args.report_html).write_text(
                _render_corpus_report_html(summary, profile_path=args.profile, mode=args.mode),
                encoding="utf-8"
            )
        if args.repro_md:
            profile = load_reference_profile(args.profile)
            algo_registry = build_algorithm_registry(
                analysis_lock=profile.analysis_lock or {},
                smoothing_cfg=profile.thresholds.get("_smoothing", {"type": "none"}),
                channel_policy=str(profile.analysis_lock.get("channel_policy", "mono"))
            )
            algo_ids = algorithm_ids_from_registry(algo_registry)
            Path(args.repro_md).write_text(
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
        help="Output path for batch summary JSON"
    )
    batch_parser.add_argument(
        "--summary-md",
        help="Output path for batch summary Markdown"
    )
    batch_parser.add_argument(
        "--report-md",
        help="Output path for one-page batch report Markdown"
    )
    batch_parser.add_argument(
        "--report-html",
        help="Output path for one-page batch report HTML"
    )
    batch_parser.add_argument(
        "--repro-md",
        help="Output path for reproducibility Markdown"
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
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(EXIT_BAD_ARGS)


if __name__ == "__main__":
    main()
