from __future__ import annotations

import numpy as np

from spectraqc.metrics.smoothing import smooth_octave_fraction, smooth_log_hz


def test_smooth_octave_fraction_preserves_constant():
    freqs = np.array([20, 40, 80, 160, 320], dtype=np.float64)
    y = np.full_like(freqs, 3.0, dtype=np.float64)
    out = smooth_octave_fraction(freqs, y, octave_fraction=1 / 3)
    assert np.allclose(out, y)


def test_smooth_log_hz_bins():
    freqs = np.array([20, 25, 30, 40], dtype=np.float64)
    y = np.array([1.0, 3.0, 5.0, 9.0], dtype=np.float64)
    out = smooth_log_hz(freqs, y, bins_per_octave=1, min_hz=20.0)
    expected_first_bin = np.mean([1.0, 3.0, 5.0])
    assert np.isclose(out[0], expected_first_bin)
    assert np.isclose(out[1], expected_first_bin)
    assert np.isclose(out[2], expected_first_bin)
    assert np.isclose(out[3], 9.0)
