from __future__ import annotations

import numpy as np

from spectraqc.dsp.psd import welch_psd_db


def test_welch_psd_db_pads_short_signal():
    x = np.random.default_rng(0).standard_normal(100)
    freqs, mean_db, var_db2 = welch_psd_db(x, fs=48000, nfft=256, hop=128)
    assert freqs.shape == mean_db.shape == var_db2.shape
    assert freqs.size == 129
