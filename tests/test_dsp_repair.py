from __future__ import annotations

import shutil

import numpy as np
import pytest

from spectraqc.dsp.repair import (
    declick,
    declip,
    dehum,
    denoise,
    loudness_normalize,
    noise_floor_dbfs,
    true_peak_limit,
)
from spectraqc.metrics.loudness import integrated_lufs_mono
from spectraqc.metrics.truepeak import true_peak_dbtp_mono


def _fft_mag_at(freq: float, x: np.ndarray, fs: float) -> float:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return float(np.abs(spectrum[idx]))


def test_dehum_reduces_hum_component():
    fs = 48000
    t = np.arange(0, 1.0, 1.0 / fs)
    hum = 0.4 * np.sin(2 * np.pi * 60.0 * t)
    tone = 0.1 * np.sin(2 * np.pi * 1000.0 * t)
    x = hum + tone
    before = _fft_mag_at(60.0, x, fs)
    out, _ = dehum(x, fs, hum_freq_hz=60.0, harmonics=1, bandwidth_hz=1.0)
    after = _fft_mag_at(60.0, out, fs)
    assert out.size == x.size
    assert after < before * 0.2


def test_declick_removes_spike():
    fs = 48000
    t = np.arange(0, 0.5, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    x[100] = 5.0
    out, metrics = declick(x, fs, threshold_sigma=4.0, window_ms=1.0)
    assert out.size == x.size
    assert metrics["clicks_fixed"] >= 1
    assert abs(out[100]) < 1.0


def test_denoise_lowers_noise_floor():
    fs = 48000
    rng = np.random.default_rng(42)
    x = 0.02 * rng.standard_normal(int(fs * 1.0))
    before = noise_floor_dbfs(x, fs)
    out, metrics = denoise(x, fs, attenuation_db=12.0, threshold_db_offset=1.0)
    after = noise_floor_dbfs(out, fs)
    assert out.size == x.size
    assert metrics["frames_attenuated"] > 0
    assert after < before


def test_declip_reduces_clipped_samples():
    fs = 48000
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 1.2 * np.sin(2 * np.pi * 1000.0 * t)
    x = np.clip(x, -1.0, 1.0)
    clipped_before = int(np.sum(np.abs(x) >= 0.98))
    out, metrics = declip(x, fs, clip_threshold=0.98)
    clipped_after = int(np.sum(np.abs(out) >= 0.98))
    assert out.size == x.size
    assert metrics["clipped_samples"] == clipped_before
    assert clipped_after < clipped_before


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_loudness_normalize_targets_lufs():
    fs = 48000
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.01 * np.sin(2 * np.pi * 1000.0 * t)
    before = integrated_lufs_mono(x, fs)
    out, metrics = loudness_normalize(x, fs, target_lufs_i=-24.0)
    after = integrated_lufs_mono(out, fs)
    assert out.size == x.size
    assert metrics["applied"] is True
    assert abs(after + 24.0) < abs(before + 24.0)


def test_true_peak_limit_reduces_peak():
    fs = 48000
    t = np.arange(0, 0.5, 1.0 / fs)
    x = 0.99 * np.sin(2 * np.pi * 1000.0 * t)
    before = true_peak_dbtp_mono(x, fs)
    out, metrics = true_peak_limit(x, fs, max_dbtp=-1.0)
    after = true_peak_dbtp_mono(out, fs)
    assert out.size == x.size
    assert metrics["applied"] is True
    assert after <= -0.8
    assert after < before
