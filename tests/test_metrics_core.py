from __future__ import annotations

import numpy as np

from spectraqc.metrics.grid import interp_to_grid, interp_var_ratio
from spectraqc.metrics.integration import band_metrics
from spectraqc.metrics.tilt import spectral_tilt_db_per_oct
from spectraqc.types import FrequencyBand


def test_interp_to_grid_clamps_edges():
    src_f = np.array([0.0, 10.0, 20.0])
    src_y = np.array([1.0, 2.0, 3.0])
    dst_f = np.array([-5.0, 0.0, 5.0, 20.0, 30.0])
    out = interp_to_grid(src_f, src_y, dst_f)
    assert np.allclose(out, [1.0, 1.0, 1.5, 3.0, 3.0])


def test_interp_var_ratio_non_negative():
    src_f = np.array([0.0, 10.0])
    src_y = np.array([-1.0, 2.0])
    dst_f = np.array([0.0, 5.0, 10.0])
    out = interp_var_ratio(src_f, src_y, dst_f)
    assert np.all(out >= 0.0)


def test_band_metrics_weighted():
    freqs = np.array([100, 200, 300, 400], dtype=np.float64)
    delta = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    var = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    ref_var = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    bands = [FrequencyBand("band1", 100, 350)]
    metrics = band_metrics(freqs, delta, var, ref_var, bands)
    assert len(metrics) == 1
    m = metrics[0]
    assert np.isclose(m.mean_deviation_db, (1.0 - 1.0 + 2.0) / 3.0)
    assert np.isclose(m.max_deviation_db, 2.0)
    assert np.isclose(m.variance_ratio, (1.0 + 1.0 + 2.0) / 3.0)


def test_spectral_tilt_recovers_slope():
    freqs = np.linspace(50.0, 16000.0, 200)
    slope = 1.5
    mean_db = slope * np.log2(freqs)
    out = spectral_tilt_db_per_oct(freqs, mean_db)
    assert np.isclose(out, slope, atol=1e-2)
