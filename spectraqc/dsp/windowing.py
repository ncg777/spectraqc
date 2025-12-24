"""Windowing functions for DSP operations."""
import numpy as np


def hann(n: int) -> np.ndarray:
    """Generate a Hann window of length n."""
    return np.hanning(n).astype(np.float64)


def window_power_norm(w: np.ndarray) -> float:
    """Compute window power normalization factor U = mean(w²)."""
    return float(np.mean(w.astype(np.float64) ** 2))
