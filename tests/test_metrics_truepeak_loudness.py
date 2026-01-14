from __future__ import annotations

import shutil
import numpy as np
import pytest

from spectraqc.metrics.truepeak import true_peak_dbtp_mono
from spectraqc.metrics.loudness import integrated_lufs_mono


def test_true_peak_basic_sine():
    fs = 48000
    t = np.arange(0, 0.1, 1.0 / fs)
    x = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    tp = true_peak_dbtp_mono(x, fs, oversample=4)
    assert np.isclose(tp, -6.02, atol=0.6)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_integrated_lufs_runs():
    fs = 48000
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.01 * np.sin(2.0 * np.pi * 1000.0 * t)
    val = integrated_lufs_mono(x, fs)
    assert np.isfinite(val)
