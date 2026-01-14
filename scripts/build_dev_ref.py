#!/usr/bin/env python
"""
Build development reference profile for SpectraQC.

Creates a reference profile from synthetic pink noise for testing.
"""
from __future__ import annotations
import argparse
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectraqc.profiles.builder import build_reference_profile_from_manifest


def main():
    """Build dev reference profile from a corpus manifest."""
    parser = argparse.ArgumentParser(description="Build reference profile from corpus manifest.")
    parser.add_argument("--manifest", required=True, help="Path to corpus manifest JSON")
    parser.add_argument("--out", help="Output path for reference profile JSON")
    parser.add_argument("--name", default="streaming_generic_v1", help="Profile name")
    parser.add_argument("--kind", default="streaming", help="Profile kind")
    args = parser.parse_args()

    try:
        profile, output_path = build_reference_profile_from_manifest(
            args.manifest,
            profile_name=args.name,
            profile_kind=args.kind,
            output_path=args.out,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Created profile: {output_path}")
    print(f"  Frequency bins: {len(profile['frequency_grid']['freqs_hz'])}")
    print(f"  Bands: {len(profile['bands'])}")
    print(f"  Hash: {profile['integrity']['profile_hash_sha256'][:16]}...")

    return 0


if __name__ == "__main__":
    exit(main())
