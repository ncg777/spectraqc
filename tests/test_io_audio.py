from __future__ import annotations

import numpy as np
import soundfile as sf

from spectraqc.io.audio import load_audio, apply_channel_policy
from spectraqc.types import AudioBuffer


def test_load_audio_and_policies(tmp_path):
    fs = 48000
    t = np.arange(0, 0.1, 1.0 / fs)
    mono = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono * 0.5], axis=1)
    path = tmp_path / "tone.wav"
    sf.write(path, stereo, fs)

    audio = load_audio(str(path))
    assert audio.channels == 2
    mono_buffers = apply_channel_policy(audio, "mono")
    assert len(mono_buffers) == 1
    assert mono_buffers[0].channels == 1
    per_channel = apply_channel_policy(audio, "per_channel")
    assert len(per_channel) == 2


def test_apply_channel_policy_mid_only():
    samples = np.array([[1.0, -1.0], [0.5, 0.5]])
    audio = AudioBuffer(samples=samples, fs=48000, duration=2 / 48000, channels=2, backend="test")
    buffers = apply_channel_policy(audio, "mid_only")
    assert len(buffers) == 1
    assert np.allclose(buffers[0].samples, [0.0, 0.5])
