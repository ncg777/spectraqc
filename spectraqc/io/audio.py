"""Audio I/O module."""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import warnings as py_warnings
from pathlib import Path
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


def _infer_bit_depth_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(\d{1,2})", text)
    if match:
        return int(match.group(1))
    normalized = text.lower()
    if "float" in normalized:
        return 32
    if "double" in normalized:
        return 64
    return None


def _extract_flac_md5(extra_info: str | None) -> str | None:
    if not extra_info:
        return None
    match = re.search(r"md5\s*[:=]\s*([0-9a-f]{32})", extra_info, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def _pcm_bytes_from_int(data: np.ndarray, bit_depth: int) -> bytes:
    if bit_depth <= 8:
        return np.asarray(data, dtype="<i1").tobytes()
    if bit_depth <= 16:
        return np.asarray(data, dtype="<i2").tobytes()
    if bit_depth <= 24:
        raw = np.asarray(data, dtype="<i4").tobytes()
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
        return packed[:, :3].tobytes()
    if bit_depth <= 32:
        return np.asarray(data, dtype="<i4").tobytes()
    raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")


def _compute_flac_pcm_md5(path: str) -> str | None:
    import soundfile as sf
    info = sf.info(path)
    bit_depth = info.bits or _infer_bit_depth_from_text(getattr(info, "subtype", None))
    if bit_depth is None:
        return None
    if isinstance(info.subtype, str) and info.subtype.lower().startswith("float"):
        return None
    dtype: str
    if bit_depth <= 8:
        dtype = "int8"
    elif bit_depth <= 16:
        dtype = "int16"
    else:
        dtype = "int32"
    data = sf.read(path, always_2d=True, dtype=dtype)[0]
    pcm_bytes = _pcm_bytes_from_int(data, bit_depth)
    return hashlib.md5(pcm_bytes).hexdigest()


def extract_audio_checksum(path: str) -> dict:
    """Extract embedded checksums and validate against decoded audio data."""
    ext = Path(path).suffix.lower()
    if ext != ".flac":
        return {
            "algorithm": "flac_md5",
            "embedded": None,
            "computed": None,
            "status": "unsupported",
        }
    import soundfile as sf
    embedded = None
    computed = None
    error = None
    try:
        info = sf.info(path)
        embedded = _extract_flac_md5(getattr(info, "extra_info", None))
        computed = _compute_flac_pcm_md5(path)
    except Exception as exc:
        error = str(exc)
    status = "unavailable"
    if embedded and computed:
        status = "match" if embedded.lower() == computed.lower() else "mismatch"
    elif embedded and not computed:
        status = "unverified"
    elif not embedded and computed:
        status = "absent"
    if error:
        status = "error"
    payload = {
        "algorithm": "flac_md5",
        "embedded": embedded,
        "computed": computed,
        "status": status,
    }
    if error:
        payload["error"] = error
    return payload


def hash_audio_data(audio: AudioBuffer) -> str:
    """Compute deterministic SHA256 hash for analyzed audio data."""
    h = hashlib.sha256()
    h.update(str(audio.fs).encode("utf-8"))
    h.update(str(audio.channels).encode("utf-8"))
    h.update(audio.samples.tobytes())
    return h.hexdigest()


def _decode_soundfile(path: str) -> tuple[np.ndarray, float, list[str], int | None]:
    """Decode using soundfile (libsndfile)."""
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile backend not available.") from exc

    with py_warnings.catch_warnings(record=True) as w:
        py_warnings.simplefilter("always")
        data, fs = sf.read(path, always_2d=True, dtype="float64")
    warn_list = [str(wi.message) for wi in w]
    bit_depth = None
    try:
        info = sf.info(path)
        if info.frames == 0:
            warn_list.append("soundfile: file reports zero frames.")
        elif info.frames > 0 and data.shape[0] < info.frames:
            warn_list.append("soundfile: decoded fewer frames than file reports.")
        if getattr(info, "bits", None):
            bit_depth = int(info.bits)
        if bit_depth is None:
            bit_depth = _infer_bit_depth_from_text(getattr(info, "subtype", None))
        if bit_depth is None:
            bit_depth = _infer_bit_depth_from_text(getattr(info, "subtype_info", None))
    except Exception:
        pass
    return data, float(fs), warn_list, bit_depth


def _parse_ffprobe_bit_depth(stream: dict) -> int | None:
    bits = stream.get("bits_per_sample")
    if bits is not None:
        try:
            bits_int = int(bits)
            if bits_int > 0:
                return bits_int
        except (TypeError, ValueError):
            pass
    sample_fmt = stream.get("sample_fmt")
    if isinstance(sample_fmt, str):
        match = re.search(r"(\d{1,2})", sample_fmt)
        if match:
            return int(match.group(1))
        if sample_fmt.startswith("flt"):
            return 32
        if sample_fmt.startswith("dbl"):
            return 64
    return None


def _ffprobe_info(path: str) -> tuple[int, int, int | None]:
    """Return (sample_rate, channels, bit_depth) from ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found for ffmpeg backend.")
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,bits_per_sample,sample_fmt",
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
    bit_depth = _parse_ffprobe_bit_depth(stream)
    return sr, ch, bit_depth


def _decode_ffmpeg(path: str) -> tuple[np.ndarray, float, list[str], int | None]:
    """Decode using ffmpeg to raw float32 PCM."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg backend not available.")
    fs, ch, bit_depth = _ffprobe_info(path)
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
    return data.astype(np.float64), float(fs), warn_list, bit_depth


def load_audio(path: str) -> AudioBuffer:
    """
    Load an audio file and normalize to mono/stereo float64.

    Supports WAV, FLAC, AIFF via soundfile; MP3 via soundfile if available,
    otherwise falls back to ffmpeg when installed.
    """
    warnings_list: list[str] = []
    backend = "soundfile"
    bit_depth = None
    try:
        data, fs, warn_list, bit_depth = _decode_soundfile(path)
        warnings_list.extend(warn_list)
    except Exception as exc:
        warnings_list.append(f"soundfile decode failed: {exc}")
        backend = "ffmpeg"
        data, fs, warn_list, bit_depth = _decode_ffmpeg(path)
        warnings_list.extend(warn_list)

    data, channels = _normalize_channels(
        data, backend=backend, warnings=warnings_list
    )
    duration = data.shape[0] / float(fs)
    if data.shape[0] == 0 or duration <= 0:
        warnings_list.append("decode produced zero-length audio.")
    return AudioBuffer(
        samples=data,
        fs=float(fs),
        duration=duration,
        channels=channels,
        backend=backend,
        bit_depth=bit_depth,
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
        bit_depth=audio.bit_depth,
        warnings=warnings_list
    )


def _mono_from_stereo(
    audio: AudioBuffer,
    *,
    policy_label: str
) -> AudioBuffer:
    """Create a mono buffer from stereo input using (L+R)/2."""
    if audio.channels == 1:
        return audio
    mono = ((audio.samples[:, 0] + audio.samples[:, 1]) * 0.5).astype(np.float64)
    warnings_list = list(audio.warnings)
    warnings_list.append(f"{audio.backend}: {policy_label} policy applied.")
    return AudioBuffer(
        samples=mono,
        fs=audio.fs,
        duration=audio.duration,
        channels=1,
        backend=audio.backend,
        bit_depth=audio.bit_depth,
        warnings=warnings_list
    )


def apply_channel_policy(audio: AudioBuffer, policy: str) -> list[AudioBuffer]:
    """Apply a channel policy and return analysis-ready buffers."""
    policy_norm = str(policy).strip().lower()
    if policy_norm == "mono":
        return [to_mono(audio)]
    if policy_norm == "stereo_average":
        return [_mono_from_stereo(audio, policy_label="stereo_average")]
    if policy_norm == "mid_only":
        return [_mono_from_stereo(audio, policy_label="mid_only")]
    if policy_norm == "per_channel":
        if audio.channels == 1:
            return [audio]
        left = AudioBuffer(
            samples=audio.samples[:, 0].astype(np.float64),
            fs=audio.fs,
            duration=audio.duration,
            channels=1,
            backend=audio.backend,
            bit_depth=audio.bit_depth,
            warnings=list(audio.warnings)
        )
        right = AudioBuffer(
            samples=audio.samples[:, 1].astype(np.float64),
            fs=audio.fs,
            duration=audio.duration,
            channels=1,
            backend=audio.backend,
            bit_depth=audio.bit_depth,
            warnings=list(audio.warnings)
        )
        return [left, right]
    raise ValueError(f"Unknown channel policy: {policy}")


def load_audio_mono(path: str) -> AudioBuffer:
    """Load audio and return a mono buffer."""
    return to_mono(load_audio(path))


# Backwards-compatible alias
load_wav_mono = load_audio_mono
