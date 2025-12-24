from __future__ import annotations
import numpy as np


def deviation_curve_db(input_mean_db: np.ndarray, ref_mean_db: np.ndarray) -> np.ndarray:
    """Compute deviation curve: input - reference in dB."""
    if input_mean_db.shape != ref_mean_db.shape:
        raise ValueError("Frequency grid mismatch between input and reference.")
    return (input_mean_db - ref_mean_db).astype(np.float64)
