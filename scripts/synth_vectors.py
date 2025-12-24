#!/usr/bin/env python
"""
Synthesize test vectors for SpectraQC validation.

Generates WAV files with known spectral characteristics for testing.
"""
from __future__ import annotations
import os
import wave
import struct
import numpy as np
from pathlib import Path


def write_wav_mono(path: str, samples: np.ndarray, fs: int = 48000) -> None:
    """Write mono samples to a 16-bit WAV file."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = np.clip(samples, -1.0, 1.0)
    samples_int = (samples * 32767).astype(np.int16)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(samples_int.tobytes())


def gen_sine(freq_hz: float, duration_s: float, fs: int, amp: float = 1.0) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(duration_s * fs)) / fs
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def gen_pink_noise(duration_s: float, fs: int, amp: float = 1.0) -> np.ndarray:
    """Generate pink noise (1/f spectrum)."""
    n_samples = int(duration_s * fs)
    # Generate white noise in frequency domain
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    # Apply 1/f weighting
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    freqs[0] = 1.0  # Avoid division by zero
    X = X / np.sqrt(freqs)
    # Back to time domain
    pink = np.fft.irfft(X, n=n_samples)
    # Normalize
    pink = pink / (np.max(np.abs(pink)) + 1e-10)
    return amp * pink


def gen_white_noise(duration_s: float, fs: int, amp: float = 1.0) -> np.ndarray:
    """Generate white noise."""
    n_samples = int(duration_s * fs)
    white = np.random.randn(n_samples)
    white = white / (np.max(np.abs(white)) + 1e-10)
    return amp * white


def gen_multitone(freqs_hz: list[float], duration_s: float, fs: int, amp: float = 1.0) -> np.ndarray:
    """Generate sum of sine waves at specified frequencies."""
    t = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(t)
    for f in freqs_hz:
        signal += np.sin(2.0 * np.pi * f * t)
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return amp * signal


def db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude."""
    return 10.0 ** (db / 20.0)


def main():
    """Generate all test vectors."""
    base_dir = Path(__file__).parent.parent / "validation" / "vectors"
    fs = 48000
    duration = 5.0  # 5 seconds
    
    print("Generating test vectors...")
    
    # Vector 1: 1kHz sine at -18 dBFS
    v1_dir = base_dir / "v0001_sine_1khz_-18dbfs"
    amp = db_to_linear(-18.0)
    samples = gen_sine(1000.0, duration, fs, amp)
    write_wav_mono(str(v1_dir / "input.wav"), samples, fs)
    print(f"  Created: {v1_dir / 'input.wav'}")
    
    # Vector 2: Pink noise at -20 dBFS
    v2_dir = base_dir / "v0002_pink_noise_-20dbfs"
    np.random.seed(42)  # Reproducible
    amp = db_to_linear(-20.0)
    samples = gen_pink_noise(duration, fs, amp)
    write_wav_mono(str(v2_dir / "input.wav"), samples, fs)
    print(f"  Created: {v2_dir / 'input.wav'}")
    
    # Vector 3: White noise at -20 dBFS
    v3_dir = base_dir / "v0003_white_noise_-20dbfs"
    np.random.seed(43)
    amp = db_to_linear(-20.0)
    samples = gen_white_noise(duration, fs, amp)
    write_wav_mono(str(v3_dir / "input.wav"), samples, fs)
    print(f"  Created: {v3_dir / 'input.wav'}")
    
    # Vector 4: Multitone (100Hz, 1kHz, 10kHz) at -12 dBFS
    v4_dir = base_dir / "v0004_multitone_-12dbfs"
    amp = db_to_linear(-12.0)
    samples = gen_multitone([100.0, 1000.0, 10000.0], duration, fs, amp)
    write_wav_mono(str(v4_dir / "input.wav"), samples, fs)
    print(f"  Created: {v4_dir / 'input.wav'}")
    
    # Vector 5: Silence (very low level noise)
    v5_dir = base_dir / "v0005_silence"
    np.random.seed(44)
    samples = gen_white_noise(duration, fs, 1e-6)
    write_wav_mono(str(v5_dir / "input.wav"), samples, fs)
    print(f"  Created: {v5_dir / 'input.wav'}")
    
    # Vector 6: Full scale sine (0 dBFS peak) for true peak testing
    v6_dir = base_dir / "v0006_fullscale_1khz"
    samples = gen_sine(1000.0, duration, fs, 0.99)
    write_wav_mono(str(v6_dir / "input.wav"), samples, fs)
    print(f"  Created: {v6_dir / 'input.wav'}")
    
    print(f"\nGenerated 6 test vectors in: {base_dir}")


if __name__ == "__main__":
    main()
