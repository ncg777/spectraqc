"""Loudness measurement module."""
from __future__ import annotations
import numpy as np
import re
import shutil
import subprocess

_LUFS_RE = re.compile(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
_LRA_RE = re.compile(r"LRA:\s*(-?\d+(?:\.\d+)?)\s*LU")


def _run_ebur128(x: np.ndarray, fs: float) -> str:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("ebur128 expects 1D mono audio.")
    if x.size == 0:
        raise ValueError("ebur128 expects non-empty audio.")

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

    return proc.stderr.decode("utf-8", errors="replace")


def _parse_integrated_lufs(stderr: str) -> float:
    matches = _LUFS_RE.findall(stderr)
    if not matches:
        raise ValueError("ffmpeg ebur128 did not report integrated loudness.")
    return float(matches[-1])


def _parse_lra(stderr: str) -> float:
    matches = _LRA_RE.findall(stderr)
    if not matches:
        raise ValueError("ffmpeg ebur128 did not report loudness range.")
    return float(matches[-1])


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
    stderr = _run_ebur128(x, fs)
    return _parse_integrated_lufs(stderr)


def loudness_range_lu_mono(x: np.ndarray, fs: float) -> float:
    """
    Compute loudness range (LRA) in LU for mono audio using ffmpeg ebur128.

    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz

    Returns:
        Loudness range in LU
    """
    stderr = _run_ebur128(x, fs)
    return _parse_lra(stderr)


def loudness_metrics_mono(x: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Compute integrated loudness (LUFS) and loudness range (LRA) for mono audio.

    Args:
        x: Mono audio samples (1D array)
        fs: Sample rate in Hz

    Returns:
        (integrated_lufs_i, loudness_range_lu)
    """
    stderr = _run_ebur128(x, fs)
    return _parse_integrated_lufs(stderr), _parse_lra(stderr)
