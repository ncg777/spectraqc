#!/usr/bin/env python
"""
Build development reference profile for SpectraQC.

Creates a reference profile from synthetic pink noise for testing.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectraqc.io.audio import load_audio_mono
from spectraqc.analysis.ltpsd import compute_ltpsd
from spectraqc.analysis.bands import default_streaming_bands
from spectraqc.metrics.smoothing import smooth_octave_fraction
from spectraqc.utils.hashing import sha256_hex_canonical_json


def build_reference_profile(
    ref_audio_path: str,
    profile_name: str = "streaming_generic_v1",
    profile_kind: str = "streaming",
) -> dict:
    """
    Build a reference profile from an audio file.
    
    Args:
        ref_audio_path: Path to reference audio file
        profile_name: Name for the profile
        profile_kind: Kind of profile (broadcast, streaming, archive, custom)
        
    Returns:
        Complete reference profile dictionary
    """
    # Load reference audio
    audio = load_audio_mono(ref_audio_path)
    
    # Compute long-term PSD
    nfft = 4096
    hop = 2048
    ltpsd = compute_ltpsd(audio, nfft=nfft, hop=hop)
    
    # Apply smoothing
    octave_fraction = 1/6
    smoothed_mean = smooth_octave_fraction(
        ltpsd.freqs, ltpsd.mean_db, octave_fraction
    )
    smoothed_var = smooth_octave_fraction(
        ltpsd.freqs, ltpsd.var_db2, octave_fraction
    )
    
    # Get bands
    bands = default_streaming_bands()
    
    # Build profile structure
    profile = {
        "profile": {
            "name": profile_name,
            "kind": profile_kind,
            "version": "1.0.0"
        },
        "frequency_grid": {
            "grid_kind": "fft_one_sided",
            "units": "Hz",
            "freqs_hz": [float(f) for f in ltpsd.freqs.tolist()]
        },
        "reference_curves": {
            "mean_db": [float(x) for x in smoothed_mean.tolist()],
            "var_db2": [float(x) for x in smoothed_var.tolist()]
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
            "normalization": {
                "loudness": {
                    "enabled": False,
                    "target_lufs_i": -14.0,
                    "algorithm_id": "bs1770-pyloudnorm"
                },
                "true_peak": {
                    "enabled": True,
                    "max_dbtp": -1.0,
                    "algorithm_id": "os4-sinc-fir"
                }
            }
        },
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
    """Build dev reference profile from pink noise vector."""
    base_dir = Path(__file__).parent.parent
    vectors_dir = base_dir / "validation" / "vectors"
    profiles_dir = base_dir / "validation" / "profiles"
    
    # Use pink noise as reference
    ref_audio = vectors_dir / "v0002_pink_noise_-20dbfs" / "input.wav"
    
    if not ref_audio.exists():
        print(f"Error: Reference audio not found at {ref_audio}")
        print("Run synth_vectors.py first to generate test vectors.")
        return 1
    
    print(f"Building reference profile from: {ref_audio}")
    
    profile = build_reference_profile(
        str(ref_audio),
        profile_name="streaming_generic_v1",
        profile_kind="streaming"
    )
    
    # Save profile
    profiles_dir.mkdir(parents=True, exist_ok=True)
    output_path = profiles_dir / "streaming_generic_v1.ref.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    
    print(f"Created profile: {output_path}")
    print(f"  Frequency bins: {len(profile['frequency_grid']['freqs_hz'])}")
    print(f"  Bands: {len(profile['bands'])}")
    print(f"  Hash: {profile['integrity']['profile_hash_sha256'][:16]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
