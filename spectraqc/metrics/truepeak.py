"""True peak measurement module."""
from __future__ import annotations
import re
import shutil
import subprocess
import numpy as np


def true_peak_dbtp_mono(
    x: np.ndarray,
    fs: float,
    oversample: int = 4
) -> float:
    """
    Compute true peak level in dBTP for mono audio via ffmpeg ebur128.

    Uses libebur128 (BS.1770-4) true-peak with 4x oversampling.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("true_peak_dbtp_mono expects 1D mono audio.")
    if x.size == 0:
        raise ValueError("true_peak_dbtp_mono expects non-empty audio.")
    if oversample != 4:
        raise ValueError("ffmpeg ebur128 true peak uses 4x oversampling only.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found for ebur128 true peak.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-v", "info",
        "-f", "f64le",
        "-ac", "1",
        "-ar", str(int(fs)),
        "-i", "pipe:0",
        "-filter_complex", "ebur128=peak=true:framelog=verbose",
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
        raise ValueError(f"ffmpeg ebur128 true peak failed: {stderr.strip()}")

    stderr = proc.stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"TP:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr)
    if not matches:
        matches = re.findall(r"True\s*peak:\s*(-?\d+(?:\.\d+)?)\s*dB", stderr, re.IGNORECASE)
    if not matches:
        raise ValueError("ffmpeg ebur128 did not report true peak.")
    return float(matches[-1])
