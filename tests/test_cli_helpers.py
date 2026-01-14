from __future__ import annotations

import numpy as np

from spectraqc.cli.main import _compute_silence_ratio, _resample_linear, _build_confidence
from spectraqc.types import AudioBuffer


def test_compute_silence_ratio_basic():
    fs = 10
    x = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    ratio = _compute_silence_ratio(x, fs, min_rms_dbfs=-40.0, frame_seconds=0.2)
    assert 0.0 <= ratio <= 1.0


def test_resample_linear_length():
    x = np.linspace(0, 1, 10)
    out = _resample_linear(x, 10.0, 20.0)
    assert out.size == 20


def test_build_confidence_flags():
    audio = AudioBuffer(
        samples=np.zeros(10),
        fs=48000.0,
        duration=0.0,
        channels=1,
        backend="test",
        warnings=["decoded fewer frames than file reports."]
    )
    conf = _build_confidence(
        audio,
        effective_duration=0.0,
        silence_ratio=1.0,
        resampled=True
    )
    assert conf["status"] == "warn"
    assert "zero_length_audio" in conf["reasons"]
