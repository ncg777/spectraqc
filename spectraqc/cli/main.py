"""SpectraQC CLI - Spectral Quality Control Tool."""
from __future__ import annotations
import argparse
import json
import sys
import platform
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


def _build_confidence(audio, effective_duration: float) -> dict:
    """Build confidence assessment based on decode sanity checks."""
    reasons: list[str] = []
    if audio.samples.size == 0 or effective_duration <= 0:
        reasons.append("zero_length_audio")
    if effective_duration < MIN_EFFECTIVE_SECONDS:
        reasons.append(f"short_effective_duration<{MIN_EFFECTIVE_SECONDS}s")
    if any(
        "trimmed partial frame" in w or "decoded fewer frames" in w
        for w in audio.warnings
    ):
        reasons.append("truncated_decode")
    status = "pass" if not reasons else "warn"
    return {
        "status": status,
        "reasons": reasons,
        "downgraded": bool(reasons)
    }


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
    
    smoothing_cfg = profile.thresholds.get("_smoothing", {"type": "none"})
    ref_var = profile.thresholds.get("_ref_var_db2", np.ones_like(profile.freqs_hz))
    ref_tilt = spectral_tilt_db_per_oct(profile.freqs_hz, profile.ref_mean_db)
    channel_policy = str(analysis_lock.get("channel_policy", "mono")).strip().lower()
    if channel_policy == "per_channel" and mode != "exploratory":
        raise ValueError("per_channel policy is only supported in exploratory mode.")
    analysis_buffers = apply_channel_policy(audio, channel_policy)

    def _analyze_single(mono_audio):
        ltpsd = compute_ltpsd(mono_audio, nfft=nfft, hop=hop)
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

        tp_dbtp = true_peak_dbtp_mono(mono_audio.samples, mono_audio.fs)

        try:
            lufs_i = integrated_lufs_mono(mono_audio.samples, mono_audio.fs)
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
        "algorithm_ids": profile.algorithm_ids
    }
    
    # Analysis configuration for report
    normalization_cfg = profile.normalization or {}
    loud_cfg = normalization_cfg.get("loudness", {})
    tp_cfg = normalization_cfg.get("true_peak", {})
    true_peak_algo_id = "bs1770-4-ffmpeg-ebur128-tp4x"
    loudness_algo_id = "bs1770-4-ffmpeg-ebur128"
    analysis_cfg = {
        "report_id": report_id,
        "created_utc": now_utc,
        "mode": mode,
        "resampled_fs_hz": audio.fs,
        "channel_policy": str(channel_policy),
        "fft_size": nfft,
        "hop_size": hop,
        "window": "hann",
        "psd_estimator": "welch",
        "smoothing": smoothing_cfg,
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
                "algorithm_id": loudness_algo_id
            },
            "true_peak": {
                "enabled": bool(tp_cfg.get("enabled", False)),
                "max_dbtp": float(tp_cfg.get("max_dbtp", -1.0)),
                "measured_dbtp": tp_dbtp,
                "algorithm_id": true_peak_algo_id
            }
        },
        "silence_gate": {
            "enabled": False,
            "min_rms_dbfs": -60.0,
            "silence_ratio": 0.0,
            "effective_seconds": audio.duration
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
        "band_decisions": [
            {
                "band_name": bd.band.name,
                "mean": {
                    "metric": bd.mean.metric,
                    "value": bd.mean.value,
                    "units": bd.mean.units,
                    "status": bd.mean.status.value,
                    "pass_limit": bd.mean.pass_limit,
                    "warn_limit": bd.mean.warn_limit
                },
                "max": {
                    "metric": bd.max.metric,
                    "value": bd.max.value,
                    "units": bd.max.units,
                    "status": bd.max.status.value,
                    "pass_limit": bd.max.pass_limit,
                    "warn_limit": bd.max.warn_limit
                },
                "variance": {
                    "metric": bd.variance.metric,
                    "value": bd.variance.value,
                    "units": bd.variance.units,
                    "status": bd.variance.status.value,
                    "pass_limit": bd.variance.pass_limit,
                    "warn_limit": bd.variance.warn_limit
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
                "warn_limit": gd.warn_limit
            }
            for gd in decision.global_decisions
        ]
    }
    
    # Confidence assessment
    confidence = _build_confidence(audio, audio.duration)
    
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
    
    return qcreport, decision, profile


def cmd_analyze(args) -> int:
    """Handle analyze command."""
    try:
        qcreport, decision, _ = _analyze_audio(
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
        _, decision, profile = _analyze_audio(
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
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(EXIT_BAD_ARGS)


if __name__ == "__main__":
    main()
