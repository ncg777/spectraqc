"""Audio I/O module."""
from __future__ import annotations
import wave
import numpy as np
from spectraqc.types import AudioBuffer


def load_wav_mono(path: str) -> AudioBuffer:
    """
    Load a WAV file and convert to mono float64.
    
    Supports 16-bit and 32-bit integer PCM formats.
    Multi-channel files are averaged to mono.
    
    Args:
        path: Path to WAV file
        
    Returns:
        AudioBuffer with mono samples normalized to [-1, 1]
        
    Raises:
        ValueError: If sample width is not supported
    """
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    x = x.reshape(-1, n_ch)
    mono = x.mean(axis=1)
    dur = mono.shape[0] / float(fs)
    
    return AudioBuffer(
        samples=mono.astype(np.float64),
        fs=float(fs),
        duration=dur
    )
