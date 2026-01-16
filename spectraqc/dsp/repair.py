"""Audio repair utilities with composable DSP steps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spectraqc.metrics.deviation import deviation_curve_db
from spectraqc.metrics.truepeak import true_peak_dbtp_mono
from spectraqc.metrics.loudness import integrated_lufs_mono
from spectraqc.metrics.grid import interp_to_grid
from spectraqc.metrics.smoothing import smooth_octave_fraction, smooth_log_hz
from spectraqc.types import AudioBuffer, ReferenceProfile


@dataclass(frozen=True)
class RepairStep:
    """Definition of a repair step."""
    name: str
    func: Callable[..., tuple[np.ndarray, dict]]


def _resample_linear(samples: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Simple linear resampling for mono buffers."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x
    if fs <= 0 or target_fs <= 0:
        raise ValueError("Sample rates must be positive.")
    if fs == target_fs:
        return x
    n_out = max(1, int(round(x.size * target_fs / fs)))
    t_in = np.arange(x.size, dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / target_fs
    return np.interp(t_out, t_in, x).astype(np.float64)


def _mono_view(samples: np.ndarray) -> np.ndarray:
    """Return mono view for metrics."""
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1)


def noise_floor_dbfs(samples: np.ndarray, fs: float, frame_seconds: float = 0.1) -> float:
    """Estimate noise floor via 10th percentile of frame RMS."""
    x = _mono_view(np.asarray(samples, dtype=np.float64))
    if x.size == 0:
        return float("-inf")
    frame_len = max(1, int(round(frame_seconds * fs)))
    rms_vals = []
    for start in range(0, x.size, frame_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_vals.append(rms)
    if not rms_vals:
        return float("-inf")
    noise_rms = float(np.percentile(rms_vals, 10))
    if noise_rms <= 0:
        return float("-inf")
    return float(20.0 * np.log10(noise_rms))


def compute_deviation_curve(
    samples: np.ndarray,
    fs: float,
    profile: ReferenceProfile
) -> np.ndarray:
    """Compute deviation curve for a mono buffer using profile grid."""
    # Local import avoids circular dependency during package initialization.
    from spectraqc.analysis.ltpsd import compute_ltpsd

    x = _mono_view(samples)
    analysis_lock = profile.analysis_lock or {}
    nfft = int(analysis_lock.get("fft_size", 4096))
    hop = int(analysis_lock.get("hop_size", max(1, nfft // 2)))
    target_fs = analysis_lock.get("resample_fs_hz")
    target_fs = float(target_fs) if target_fs is not None else None
    smoothing_cfg = profile.thresholds.get("_smoothing", {"type": "none"})
    if target_fs is not None and target_fs > 0 and target_fs != fs:
        x = _resample_linear(x, fs, target_fs)
        fs = target_fs
    buffer = AudioBuffer(
        samples=x.astype(np.float64),
        fs=float(fs),
        duration=x.size / fs if fs > 0 else 0.0,
        channels=1,
        backend="repair",
        warnings=[]
    )
    ltpsd = compute_ltpsd(buffer, nfft=nfft, hop=hop)
    input_mean_db = interp_to_grid(ltpsd.freqs, ltpsd.mean_db, profile.freqs_hz)
    if smoothing_cfg.get("type") == "octave_fraction":
        oct_frac = smoothing_cfg.get("octave_fraction", 1 / 6)
        input_mean_db = smooth_octave_fraction(profile.freqs_hz, input_mean_db, oct_frac)
    elif smoothing_cfg.get("type") == "log_hz":
        bins_per_oct = smoothing_cfg.get("log_hz_bins_per_octave", 12)
        input_mean_db = smooth_log_hz(
            profile.freqs_hz,
            input_mean_db,
            bins_per_octave=int(bins_per_oct)
        )
    return deviation_curve_db(input_mean_db, profile.ref_mean_db)


def _apply_per_channel(
    samples: np.ndarray,
    fs: float,
    step_fn: Callable[..., tuple[np.ndarray, dict]],
    **params
) -> tuple[np.ndarray, dict]:
    """Apply a step per channel and return combined metrics."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        out, metrics = step_fn(x, fs, **params)
        return out, metrics
    outputs = []
    metrics_list = []
    for ch in range(x.shape[1]):
        out, metrics = step_fn(x[:, ch], fs, **params)
        outputs.append(out)
        metrics_list.append(metrics)
    stacked = np.stack(outputs, axis=1)
    return stacked, {"channels": metrics_list}


def dehum(
    samples: np.ndarray,
    fs: float,
    *,
    hum_freq_hz: float = 60.0,
    harmonics: int = 5,
    bandwidth_hz: float = 1.0
) -> tuple[np.ndarray, dict]:
    """Remove mains hum via frequency-domain notches."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(x)
    half_bw = max(0.1, float(bandwidth_hz)) / 2.0
    removed_bins = 0
    for h in range(1, harmonics + 1):
        target = hum_freq_hz * h
        if target <= 0 or target >= fs / 2:
            continue
        mask = (freqs >= target - half_bw) & (freqs <= target + half_bw)
        removed_bins += int(np.sum(mask))
        spectrum[mask] = 0.0
    cleaned = np.fft.irfft(spectrum, n=n).astype(np.float64)
    return cleaned, {"applied": True, "removed_bins": removed_bins}


def declick(
    samples: np.ndarray,
    fs: float,
    *,
    threshold_sigma: float = 6.0,
    window_ms: float = 1.0
) -> tuple[np.ndarray, dict]:
    """Remove impulsive clicks via median replacement."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    diffs = np.abs(np.diff(x, prepend=x[0]))
    median_diff = np.median(diffs)
    thresh = float(threshold_sigma) * (median_diff + 1e-12)
    click_idx = np.where(diffs > thresh)[0]
    if click_idx.size == 0:
        return x, {"applied": False, "clicks_fixed": 0}
    window = max(1, int(round(window_ms * 1e-3 * fs)))
    out = x.copy()
    for idx in click_idx:
        start = max(0, idx - window)
        end = min(x.size, idx + window + 1)
        out[idx] = float(np.median(x[start:end]))
    return out, {"applied": True, "clicks_fixed": int(click_idx.size)}


def denoise(
    samples: np.ndarray,
    fs: float,
    *,
    frame_seconds: float = 0.1,
    attenuation_db: float = 6.0,
    threshold_db_offset: float = 3.0
) -> tuple[np.ndarray, dict]:
    """Apply simple noise gate by attenuating low-RMS frames."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    frame_len = max(1, int(round(frame_seconds * fs)))
    rms_vals = []
    for start in range(0, x.size, frame_len):
        frame = x[start:start + frame_len]
        if frame.size == 0:
            continue
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))
    if not rms_vals:
        return x, {"applied": False, "reason": "no_frames"}
    noise_rms = float(np.percentile(rms_vals, 10))
    noise_db = 20.0 * np.log10(max(noise_rms, 1e-12))
    threshold_db = noise_db + float(threshold_db_offset)
    threshold_rms = 10.0 ** (threshold_db / 20.0)
    gain = 10.0 ** (-float(attenuation_db) / 20.0)
    out = x.copy()
    attenuated = 0
    for start in range(0, x.size, frame_len):
        end = min(x.size, start + frame_len)
        frame = out[start:end]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms <= threshold_rms:
            out[start:end] = frame * gain
            attenuated += 1
    return out, {
        "applied": True,
        "frames_attenuated": attenuated,
        "noise_floor_dbfs": noise_db
    }


def declip(
    samples: np.ndarray,
    fs: float,
    *,
    clip_threshold: float = 0.98
) -> tuple[np.ndarray, dict]:
    """Restore clipped samples by linear interpolation."""
    _ = fs
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    clipped = np.where(np.abs(x) >= clip_threshold)[0]
    if clipped.size == 0:
        return x, {"applied": False, "clipped_samples": 0}
    out = x.copy()
    mask = np.abs(x) >= clip_threshold
    idx = np.arange(x.size)
    good = ~mask
    if np.any(good):
        out[mask] = np.interp(idx[mask], idx[good], x[good])
    return out, {"applied": True, "clipped_samples": int(clipped.size)}


def loudness_normalize(
    samples: np.ndarray,
    fs: float,
    *,
    target_lufs_i: float = -24.0
) -> tuple[np.ndarray, dict]:
    """Normalize loudness using integrated LUFS measurement."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    mono = _mono_view(x)
    lufs = integrated_lufs_mono(mono, fs)
    gain_db = float(target_lufs_i) - float(lufs)
    gain = 10.0 ** (gain_db / 20.0)
    out = x * gain
    return out.astype(np.float64), {"applied": True, "measured_lufs_i": lufs, "gain_db": gain_db}


def true_peak_limit(
    samples: np.ndarray,
    fs: float,
    *,
    max_dbtp: float = -1.0
) -> tuple[np.ndarray, dict]:
    """Apply static gain to satisfy true-peak ceiling."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x, {"applied": False, "reason": "empty"}
    mono = _mono_view(x)
    tp = true_peak_dbtp_mono(mono, fs)
    if tp <= max_dbtp:
        return x, {"applied": False, "true_peak_dbtp": tp}
    gain_db = float(max_dbtp) - float(tp)
    gain = 10.0 ** (gain_db / 20.0)
    out = x * gain
    return out.astype(np.float64), {"applied": True, "true_peak_dbtp": tp, "gain_db": gain_db}


STEP_REGISTRY: dict[str, RepairStep] = {
    "dehum": RepairStep("dehum", dehum),
    "declick": RepairStep("declick", declick),
    "denoise": RepairStep("denoise", denoise),
    "declip": RepairStep("declip", declip),
    "loudness_normalize": RepairStep("loudness_normalize", loudness_normalize),
    "true_peak_limit": RepairStep("true_peak_limit", true_peak_limit),
}


def apply_repair_plan(
    samples: np.ndarray,
    fs: float,
    plan: dict
) -> tuple[np.ndarray, list[dict]]:
    """Apply a repair plan to samples and return step results."""
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("repair plan steps must be a list.")
    x = np.asarray(samples, dtype=np.float64)
    executed: list[dict] = []
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if not name:
            raise ValueError("repair plan step missing name.")
        if step.get("enabled", True) is False:
            continue
        entry = STEP_REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown repair step: {name}")
        params = dict(step.get("params", {}))
        x, metrics = _apply_per_channel(x, fs, entry.func, **params)
        executed.append({
            "name": name,
            "params": params,
            "metrics": metrics
        })
    return x, executed


def compute_repair_metrics(
    samples: np.ndarray,
    fs: float,
    profile: ReferenceProfile
) -> dict:
    """Compute key metrics for repair reporting."""
    mono = _mono_view(np.asarray(samples, dtype=np.float64))
    metrics = {
        "true_peak_dbtp": true_peak_dbtp_mono(mono, fs),
        "noise_floor_dbfs": noise_floor_dbfs(mono, fs),
        "deviation_curve_db": compute_deviation_curve(mono, fs, profile).tolist()
    }
    return metrics


def suggest_repair_plan(
    samples: np.ndarray,
    fs: float,
    profile: ReferenceProfile
) -> tuple[dict, dict]:
    """Suggest a repair plan based on quick diagnostics."""
    mono = _mono_view(np.asarray(samples, dtype=np.float64))
    metrics = compute_repair_metrics(mono, fs, profile)
    steps: list[dict] = []
    notes: list[str] = []

    clip_threshold = 0.98
    clip_ratio = float(np.mean(np.abs(mono) >= clip_threshold)) if mono.size else 0.0
    if clip_ratio > 0.0001:
        steps.append({"name": "declip", "params": {"clip_threshold": clip_threshold}})
        notes.append(f"Detected clipping ratio {clip_ratio:.4f}, adding declip.")

    noise_floor = metrics["noise_floor_dbfs"]
    if noise_floor > -60.0:
        steps.append({
            "name": "denoise",
            "params": {"frame_seconds": 0.1, "attenuation_db": 6.0, "threshold_db_offset": 3.0}
        })
        notes.append(f"Noise floor {noise_floor:.1f} dBFS, adding denoise.")

    deviation = np.array(metrics["deviation_curve_db"], dtype=np.float64)
    freqs = np.array(profile.freqs_hz, dtype=np.float64)
    hum_candidates = [50.0, 60.0]
    for hum in hum_candidates:
        if freqs.size == 0:
            break
        mask = (freqs >= hum - 1.0) & (freqs <= hum + 1.0)
        if np.any(mask):
            hum_dev = float(np.mean(deviation[mask]))
            if hum_dev > 6.0:
                steps.append({"name": "dehum", "params": {"hum_freq_hz": hum, "harmonics": 5, "bandwidth_hz": 1.0}})
                notes.append(f"Detected hum around {hum:.0f} Hz (+{hum_dev:.1f} dB), adding dehum.")
                break

    normalization = profile.normalization or {}
    loud_cfg = normalization.get("loudness", {})
    if loud_cfg.get("enabled", False):
        target = float(loud_cfg.get("target_lufs_i", -24.0))
        steps.append({"name": "loudness_normalize", "params": {"target_lufs_i": target}})
        notes.append(f"Loudness normalization enabled, target {target:.1f} LUFS.")

    tp_cfg = normalization.get("true_peak", {})
    max_dbtp = None
    if tp_cfg.get("enabled", False):
        max_dbtp = float(tp_cfg.get("max_dbtp", -1.0))
    if max_dbtp is not None and metrics["true_peak_dbtp"] > max_dbtp:
        steps.append({"name": "true_peak_limit", "params": {"max_dbtp": max_dbtp}})
        notes.append(
            f"True peak {metrics['true_peak_dbtp']:.2f} dBTP exceeds {max_dbtp:.2f} dBTP, adding limiter."
        )

    plan = {
        "schema_version": "1.0",
        "profile": {
            "name": profile.name,
            "kind": profile.kind,
            "version": profile.version,
        },
        "generated": {
            "notes": notes,
            "metrics": metrics,
        },
        "steps": steps,
    }
    summary = {
        "suggested_step_count": len(steps),
        "clip_ratio": clip_ratio,
        "noise_floor_dbfs": noise_floor,
        "true_peak_dbtp": metrics["true_peak_dbtp"],
        "notes": notes,
    }
    return plan, summary
