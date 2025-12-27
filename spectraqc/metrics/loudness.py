"""Loudness measurement module."""
from __future__ import annotations
import numpy as np
import re
import shutil
import subprocess


def integrated_lufs_mono(x: np.ndarray, fs: float) -> float:
    """
    Compute integrated loudness in LUFS for mono audio using ffmpeg ebur128.

    Uses BS.1770-4 reference implementation via ffmpeg's ebur128 filter.
    
    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz
        
    Returns:
        Integrated loudness in LUFS
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("integrated_lufs_mono expects 1D mono audio.")
    if x.size == 0:
        raise ValueError("integrated_lufs_mono expects non-empty audio.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found for ebur128 loudness.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-v", "info",
        "-f", "f64le",
        "-ac", "1",
        "-ar", str(int(fs)),
        "-i", "pipe:0",
        "-filter_complex", "ebur128=framelog=verbose:peak=true",
        "-f", "null",
        "-"
    ]

    proc = subprocess.run(
        cmd,
        input=x.tobytes(),
        capture_output=True,
        check=False
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"ffmpeg ebur128 failed: {stderr.strip()}")

    stderr = proc.stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", stderr)
    if not matches:
        raise ValueError("ffmpeg ebur128 did not report integrated loudness.")
    return float(matches[-1])
