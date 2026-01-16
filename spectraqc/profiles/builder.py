from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectraqc.analysis.bands import default_streaming_bands
from spectraqc.analysis.ltpsd import compute_ltpsd
from spectraqc.algorithms.registry import (
    LOUDNESS_ALGO_ID,
    TRUE_PEAK_ALGO_ID,
    algorithm_ids_from_registry,
    build_algorithm_registry,
)
from spectraqc.corpus.manifest import load_corpus_manifest, validate_manifest_entry
from spectraqc.io.audio import load_audio_mono
from spectraqc.metrics.smoothing import smooth_octave_fraction
from spectraqc.metrics.tonal import derive_noise_floor_baselines
from spectraqc.thresholds.level_metrics import DEFAULT_LEVEL_METRIC_THRESHOLDS
from spectraqc.utils.hashing import sha256_hex_canonical_json

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".mp3"}


def build_reference_profile(
    ref_audio_paths: list[str],
    *,
    profile_name: str,
    profile_kind: str,
) -> dict:
    """Build a reference profile from a list of reference audio paths."""
    nfft = 4096
    hop = 2048

    if not ref_audio_paths:
        raise ValueError("At least one reference audio path is required.")

    mean_accum = None
    var_accum = None
    weight_total = 0.0
    freqs = None
    per_file_mean = []
    per_file_var = []
    durations = []
    sample_rates = []

    for path in ref_audio_paths:
        audio = load_audio_mono(path)
        durations.append(audio.duration)
        sample_rates.append(int(audio.fs))
        ltpsd = compute_ltpsd(audio, nfft=nfft, hop=hop)
        if freqs is None:
            freqs = ltpsd.freqs
        elif ltpsd.freqs.shape != freqs.shape or not np.allclose(ltpsd.freqs, freqs):
            raise ValueError("Frequency grid mismatch across corpus.")

        weight = max(audio.duration, 1e-9)
        mean_curve = smooth_octave_fraction(freqs, ltpsd.mean_db, 1 / 6)
        var_curve = smooth_octave_fraction(freqs, ltpsd.var_db2, 1 / 6)
        per_file_mean.append(mean_curve)
        per_file_var.append(var_curve)
        if mean_accum is None:
            mean_accum = mean_curve * weight
            var_accum = var_curve * weight
        else:
            mean_accum += mean_curve * weight
            var_accum += var_curve * weight
        weight_total += weight

    mean_db = mean_accum / weight_total
    var_db2 = var_accum / weight_total
    octave_fraction = 1 / 6

    bands = default_streaming_bands()
    noise_floor_baselines = derive_noise_floor_baselines(freqs, mean_db, bands)
    algorithm_registry = build_algorithm_registry(
        analysis_lock={
            "fft_size": nfft,
            "hop_size": hop,
            "window": "hann",
            "psd_estimator": "welch",
            "channel_policy": "mono",
        },
        smoothing_cfg={"type": "octave_fraction", "octave_fraction": octave_fraction},
        channel_policy="mono",
    )
    algorithm_ids = algorithm_ids_from_registry(algorithm_registry)

    profile = {
        "schema_version": "2.0",
        "profile": {
            "name": profile_name,
            "kind": profile_kind,
            "version": "1.0.0",
        },
        "frequency_grid": {
            "grid_kind": "fft_one_sided",
            "units": "Hz",
            "freqs_hz": [float(f) for f in freqs.tolist()],
        },
        "reference_curves": {
            "mean_db": [float(x) for x in mean_db.tolist()],
            "var_db2": [float(x) for x in var_db2.tolist()],
            "mean_db_p10": [float(x) for x in np.percentile(per_file_mean, 10, axis=0).tolist()],
            "mean_db_p50": [float(x) for x in np.percentile(per_file_mean, 50, axis=0).tolist()],
            "mean_db_p90": [float(x) for x in np.percentile(per_file_mean, 90, axis=0).tolist()],
            "var_db2_p10": [float(x) for x in np.percentile(per_file_var, 10, axis=0).tolist()],
            "var_db2_p50": [float(x) for x in np.percentile(per_file_var, 50, axis=0).tolist()],
            "var_db2_p90": [float(x) for x in np.percentile(per_file_var, 90, axis=0).tolist()],
        },
        "bands": [
            {"name": b.name, "f_low_hz": b.f_low, "f_high_hz": b.f_high}
            for b in bands
        ],
        "noise_floor_baselines": [
            {"band_name": b.name, "noise_floor_db": noise_floor_baselines[b.name]}
            for b in bands
        ],
        "analysis_lock": {
            "fft_size": nfft,
            "hop_size": hop,
            "window": "hann",
            "psd_estimator": "welch",
            "smoothing": {
                "type": "octave_fraction",
                "octave_fraction": octave_fraction,
            },
            "channel_policy": "mono",
            "dynamic_range": {
                "rms_percentile": {
                    "frame_seconds": 3.0,
                    "hop_seconds": 1.0,
                    "low_percentile": 10.0,
                    "high_percentile": 95.0,
                },
                "short_term_lufs": {
                    "low_percentile": 10.0,
                    "high_percentile": 95.0,
                },
            },
            "normalization": {
                "loudness": {
                    "enabled": False,
                    "target_lufs_i": -14.0,
                    "algorithm_id": LOUDNESS_ALGO_ID,
                },
                "true_peak": {
                    "enabled": True,
                    "max_dbtp": -1.0,
                    "algorithm_id": TRUE_PEAK_ALGO_ID,
                },
            },
        },
        "corpus_stats": {
            "file_count": int(len(ref_audio_paths)),
            "total_duration_s": float(sum(durations)),
            "mean_duration_s": float(np.mean(durations)),
            "min_duration_s": float(np.min(durations)),
            "max_duration_s": float(np.max(durations)),
            "sample_rates_hz": sorted({int(sr) for sr in sample_rates}),
            "percentiles": [10, 50, 90],
            "percentile_weighting": "unweighted",
        },
        "algorithm_registry": algorithm_registry,
        "algorithm_ids": algorithm_ids,
        "threshold_model": {
            "rules": {
                "band_mean": {
                    "default": {"pass": 2.0, "warn": 4.0},
                    "by_band": [
                        {"band_name": "sub", "pass": 3.0, "warn": 6.0},
                        {"band_name": "air", "pass": 4.0, "warn": 8.0},
                    ],
                },
                "band_max": {
                    "default": {"pass": 4.0, "warn": 8.0},
                    "by_band": [],
                },
                "tilt": {"pass": 0.5, "warn": 1.0},
                "peak_dbfs": DEFAULT_LEVEL_METRIC_THRESHOLDS["peak_dbfs"],
                "true_peak_dbtp": DEFAULT_LEVEL_METRIC_THRESHOLDS["true_peak_dbtp"],
                "rms_dbfs": DEFAULT_LEVEL_METRIC_THRESHOLDS["rms_dbfs"],
                "noise_floor_dbfs": {"pass": -60.0, "warn": -54.0},
                "crest_factor_db": {"pass": 12.0, "warn": 8.0},
                "lufs_i": DEFAULT_LEVEL_METRIC_THRESHOLDS["lufs_i"],
                "loudness_range": {"pass": 12.0, "warn": 18.0},
                "dynamic_range_db": {"pass": 8.0, "warn": 6.0},
                "dynamic_range_lu": {"pass": 8.0, "warn": 6.0},
                "tonal_peak": {"pass": 6.0, "warn": 10.0},
                "level_anomalies": {
                    "channel_policy": "per_channel",
                    "drop": {
                        "frame_seconds": 0.1,
                        "hop_seconds": 0.05,
                        "baseline_window_seconds": 1.0,
                        "drop_db": 24.0,
                        "min_duration_seconds": 0.1,
                        "floor_dbfs": -120.0,
                        "warn_count": 1,
                        "fail_count": 3,
                        "warn_total_seconds": 0.1,
                        "fail_total_seconds": 0.5,
                    },
                    "zero": {
                        "zero_threshold": 1e-8,
                        "min_duration_seconds": 0.01,
                        "warn_count": 1,
                        "fail_count": 3,
                        "warn_total_seconds": 0.1,
                        "fail_total_seconds": 0.5,
                    },
                },
            },
            "aggregation": {
                "warn_if_warn_band_count_at_least": 2,
            },
        },
        "integrity": {
            "profile_hash_sha256": "",
            "signed": False,
            "signature": {"algo": "none", "value_b64": ""},
        },
    }

    tmp = dict(profile)
    tmp.pop("integrity", None)
    profile["integrity"]["profile_hash_sha256"] = sha256_hex_canonical_json(tmp)

    return profile


def build_reference_profile_from_manifest(
    manifest_path: str,
    *,
    profile_name: str,
    profile_kind: str,
    output_path: str | None = None,
) -> tuple[dict, Path]:
    """Build and write a reference profile from a corpus manifest."""
    manifest_path = str(manifest_path)
    _, entries, _ = load_corpus_manifest(manifest_path)
    ref_paths: list[str] = []
    for entry in entries:
        if entry.exclude:
            continue
        audio = load_audio_mono(str(entry.path))
        validate_manifest_entry(entry, duration_s=audio.duration)
        ref_paths.append(str(entry.path))

    if not ref_paths:
        raise ValueError("No usable entries in manifest.")

    profile = build_reference_profile(
        ref_paths,
        profile_name=profile_name,
        profile_kind=profile_kind,
    )

    base_dir = Path(__file__).parent.parent.parent
    profiles_dir = base_dir / "validation" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = Path(output_path) if output_path else profiles_dir / f"{profile_name}.ref.json"
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return profile, resolved_output


def build_reference_profile_from_folder(
    folder: str,
    *,
    recursive: bool,
    profile_name: str,
    profile_kind: str,
    output_path: str | None = None,
) -> tuple[dict, Path]:
    """Build and write a reference profile from a folder of audio files."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    files = folder_path.rglob("*") if recursive else folder_path.glob("*")
    ref_paths = [str(p) for p in files if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    if not ref_paths:
        raise ValueError("No supported audio files found in folder.")

    profile = build_reference_profile(
        ref_paths,
        profile_name=profile_name,
        profile_kind=profile_kind,
    )

    base_dir = Path(__file__).parent.parent.parent
    profiles_dir = base_dir / "validation" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = Path(output_path) if output_path else profiles_dir / f"{profile_name}.ref.json"
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return profile, resolved_output
