#!/usr/bin/env python
"""
Build development reference profile for SpectraQC.

Creates a reference profile from synthetic pink noise for testing.
"""
from __future__ import annotations
import json
import numpy as np
import argparse
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectraqc.io.audio import load_audio_mono
from spectraqc.analysis.ltpsd import compute_ltpsd
from spectraqc.analysis.bands import default_streaming_bands
from spectraqc.metrics.smoothing import smooth_octave_fraction
from spectraqc.utils.hashing import sha256_hex_canonical_json
from spectraqc.corpus.manifest import load_corpus_manifest, validate_manifest_entry
from spectraqc.algorithms.registry import (
    build_algorithm_registry,
    algorithm_ids_from_registry,
    LOUDNESS_ALGO_ID,
    TRUE_PEAK_ALGO_ID,
)


def build_reference_profile(
    ref_audio_paths: list[str],
    profile_name: str = "streaming_generic_v1",
    profile_kind: str = "streaming",
) -> dict:
    """
    Build a reference profile from a corpus of audio files.
    
    Args:
        ref_audio_paths: List of paths to reference audio files
        profile_name: Name for the profile
        profile_kind: Kind of profile (broadcast, streaming, archive, custom)
        
    Returns:
        Complete reference profile dictionary
    """
    nfft = 4096
    hop = 2048

    # Compute long-term PSD across corpus
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
    for p in ref_audio_paths:
        audio = load_audio_mono(p)
        durations.append(audio.duration)
        sample_rates.append(int(audio.fs))
        ltpsd = compute_ltpsd(audio, nfft=nfft, hop=hop)
        if freqs is None:
            freqs = ltpsd.freqs
        else:
            if ltpsd.freqs.shape != freqs.shape or not np.allclose(ltpsd.freqs, freqs):
                raise ValueError("Frequency grid mismatch across corpus.")
        weight = max(audio.duration, 1e-9)
        mean_curve = ltpsd.mean_db
        var_curve = ltpsd.var_db2
        mean_curve = smooth_octave_fraction(freqs, mean_curve, 1 / 6)
        var_curve = smooth_octave_fraction(freqs, var_curve, 1 / 6)
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
    smoothed_mean = mean_db
    smoothed_var = var_db2
    
    # Get bands
    bands = default_streaming_bands()
    
    # Build profile structure
    algorithm_registry = build_algorithm_registry(
        analysis_lock={
            "fft_size": nfft,
            "hop_size": hop,
            "window": "hann",
            "psd_estimator": "welch",
            "channel_policy": "mono"
        },
        smoothing_cfg={"type": "octave_fraction", "octave_fraction": octave_fraction},
        channel_policy="mono"
    )
    algorithm_ids = algorithm_ids_from_registry(algorithm_registry)
    profile = {
        "schema_version": "2.0",
        "profile": {
            "name": profile_name,
            "kind": profile_kind,
            "version": "1.0.0"
        },
        "frequency_grid": {
            "grid_kind": "fft_one_sided",
            "units": "Hz",
            "freqs_hz": [float(f) for f in freqs.tolist()]
        },
        "reference_curves": {
            "mean_db": [float(x) for x in smoothed_mean.tolist()],
            "var_db2": [float(x) for x in smoothed_var.tolist()],
            "mean_db_p10": [float(x) for x in np.percentile(per_file_mean, 10, axis=0).tolist()],
            "mean_db_p50": [float(x) for x in np.percentile(per_file_mean, 50, axis=0).tolist()],
            "mean_db_p90": [float(x) for x in np.percentile(per_file_mean, 90, axis=0).tolist()],
            "var_db2_p10": [float(x) for x in np.percentile(per_file_var, 10, axis=0).tolist()],
            "var_db2_p50": [float(x) for x in np.percentile(per_file_var, 50, axis=0).tolist()],
            "var_db2_p90": [float(x) for x in np.percentile(per_file_var, 90, axis=0).tolist()]
        },
        "bands": [
            {"name": b.name, "f_low_hz": b.f_low, "f_high_hz": b.f_high}
            for b in bands
        ],
        "analysis_lock": {
            "fft_size": nfft,
            "hop_size": hop,
            "window": "hann",
            "psd_estimator": "welch",
            "smoothing": {
                "type": "octave_fraction",
                "octave_fraction": octave_fraction
            },
            "channel_policy": "mono",
            "normalization": {
                "loudness": {
                    "enabled": False,
                    "target_lufs_i": -14.0,
                    "algorithm_id": LOUDNESS_ALGO_ID
                },
                "true_peak": {
                    "enabled": True,
                    "max_dbtp": -1.0,
                    "algorithm_id": TRUE_PEAK_ALGO_ID
                }
            }
        },
        "corpus_stats": {
            "file_count": int(len(ref_audio_paths)),
            "total_duration_s": float(sum(durations)),
            "mean_duration_s": float(np.mean(durations)),
            "min_duration_s": float(np.min(durations)),
            "max_duration_s": float(np.max(durations)),
            "sample_rates_hz": sorted({int(sr) for sr in sample_rates}),
            "percentiles": [10, 50, 90],
            "percentile_weighting": "unweighted"
        },
        "algorithm_registry": algorithm_registry,
        "algorithm_ids": algorithm_ids,
        "threshold_model": {
            "rules": {
                "band_mean": {
                    "default": {"pass": 2.0, "warn": 4.0},
                    "by_band": [
                        {"band_name": "sub", "pass": 3.0, "warn": 6.0},
                        {"band_name": "air", "pass": 4.0, "warn": 8.0}
                    ]
                },
                "band_max": {
                    "default": {"pass": 4.0, "warn": 8.0},
                    "by_band": []
                },
                "tilt": {"pass": 0.5, "warn": 1.0}
            },
            "aggregation": {
                "warn_if_warn_band_count_at_least": 2
            }
        },
        "integrity": {
            "profile_hash_sha256": "",
            "signed": False,
            "signature": {"algo": "none", "value_b64": ""}
        }
    }
    
    # Compute hash
    tmp = dict(profile)
    tmp.pop("integrity", None)
    profile["integrity"]["profile_hash_sha256"] = sha256_hex_canonical_json(tmp)
    
    return profile


def main():
    """Build dev reference profile from a corpus manifest."""
    parser = argparse.ArgumentParser(description="Build reference profile from corpus manifest.")
    parser.add_argument("--manifest", required=True, help="Path to corpus manifest JSON")
    parser.add_argument("--out", help="Output path for reference profile JSON")
    parser.add_argument("--name", default="streaming_generic_v1", help="Profile name")
    parser.add_argument("--kind", default="streaming", help="Profile kind")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    profiles_dir = base_dir / "validation" / "profiles"

    _, entries, _ = load_corpus_manifest(args.manifest)
    ref_paths: list[str] = []
    for e in entries:
        if e.exclude:
            continue
        audio = load_audio_mono(str(e.path))
        validate_manifest_entry(e, duration_s=audio.duration)
        ref_paths.append(str(e.path))

    if not ref_paths:
        print("Error: No usable entries in manifest.")
        return 1

    profile = build_reference_profile(
        ref_paths,
        profile_name=args.name,
        profile_kind=args.kind
    )

    profiles_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else profiles_dir / f"{args.name}.ref.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Created profile: {output_path}")
    print(f"  Frequency bins: {len(profile['frequency_grid']['freqs_hz'])}")
    print(f"  Bands: {len(profile['bands'])}")
    print(f"  Hash: {profile['integrity']['profile_hash_sha256'][:16]}...")

    return 0


if __name__ == "__main__":
    exit(main())
