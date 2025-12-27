"""Audio I/O module."""
from __future__ import annotations
import json
import shutil
import subprocess
import warnings as py_warnings
import numpy as np
from spectraqc.types import AudioBuffer


def _normalize_channels(
    samples: np.ndarray,
    *,
    backend: str,
    warnings: list[str]
) -> tuple[np.ndarray, int]:
    """Normalize decoded audio to mono or stereo float64 buffers."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        return x, 1
    if x.ndim != 2:
        raise ValueError("Decoded audio must be 1D or 2D array.")
    if x.shape[1] == 1:
        return x[:, 0], 1
    if x.shape[1] == 2:
        return x, 2
    warnings.append(
        f"{backend}: downmixed {x.shape[1]} channels to mono."
    )
    return np.mean(x, axis=1), 1


def _decode_soundfile(path: str) -> tuple[np.ndarray, float, list[str]]:
    """Decode using soundfile (libsndfile)."""
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile backend not available.") from exc

    with py_warnings.catch_warnings(record=True) as w:
        py_warnings.simplefilter("always")
        data, fs = sf.read(path, always_2d=True, dtype="float64")
    warn_list = [str(wi.message) for wi in w]
    return data, float(fs), warn_list


def _ffprobe_info(path: str) -> tuple[int, int]:
    """Return (sample_rate, channels) from ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found for ffmpeg backend.")
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"ffprobe failed: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise ValueError("ffprobe reported no audio streams.")
    stream = streams[0]
    sr = int(stream["sample_rate"])
    ch = int(stream["channels"])
    return sr, ch


def _decode_ffmpeg(path: str) -> tuple[np.ndarray, float, list[str]]:
    """Decode using ffmpeg to raw float32 PCM."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg backend not available.")
    fs, ch = _ffprobe_info(path)
    cmd = [
        ffmpeg,
        "-v", "warning",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-vn",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    warn_list = [line for line in proc.stderr.decode("utf-8", errors="replace").splitlines() if line.strip()]
    if proc.returncode != 0:
        raise ValueError("ffmpeg decode failed.")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if ch > 0:
        n = (data.size // ch) * ch
        if n != data.size:
            warn_list.append("ffmpeg: trimmed partial frame at end of stream.")
            data = data[:n]
        data = data.reshape(-1, ch)
    return data.astype(np.float64), float(fs), warn_list


def load_audio(path: str) -> AudioBuffer:
    """
    Load an audio file and normalize to mono/stereo float64.

    Supports WAV, FLAC, AIFF via soundfile; MP3 via soundfile if available,
    otherwise falls back to ffmpeg when installed.
    """
    warnings_list: list[str] = []
    backend = "soundfile"
    try:
        data, fs, warn_list = _decode_soundfile(path)
        warnings_list.extend(warn_list)
    except Exception as exc:
        warnings_list.append(f"soundfile decode failed: {exc}")
        backend = "ffmpeg"
        data, fs, warn_list = _decode_ffmpeg(path)
        warnings_list.extend(warn_list)

    data, channels = _normalize_channels(
        data, backend=backend, warnings=warnings_list
    )
    duration = data.shape[0] / float(fs)
    return AudioBuffer(
        samples=data,
        fs=float(fs),
        duration=duration,
        channels=channels,
        backend=backend,
        warnings=warnings_list
    )


def to_mono(audio: AudioBuffer) -> AudioBuffer:
    """Downmix an AudioBuffer to mono if needed."""
    if audio.channels == 1:
        return audio
    mono = np.mean(audio.samples, axis=1).astype(np.float64)
    warnings_list = list(audio.warnings)
    warnings_list.append(f"{audio.backend}: downmixed stereo to mono.")
    return AudioBuffer(
        samples=mono,
        fs=audio.fs,
        duration=audio.duration,
        channels=1,
        backend=audio.backend,
        warnings=warnings_list
    )


def load_audio_mono(path: str) -> AudioBuffer:
    """Load audio and return a mono buffer."""
    return to_mono(load_audio(path))


# Backwards-compatible alias
load_wav_mono = load_audio_mono
