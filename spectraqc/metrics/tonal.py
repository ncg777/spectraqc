from __future__ import annotations

from typing import Iterable

import numpy as np

from spectraqc.types import FrequencyBand


DEFAULT_TONAL_DETECTION = {
    "min_freq_hz": 20.0,
    "max_freq_hz": None,
    "min_prominence_db": 6.0,
    "neighborhood_bins": 3,
    "harmonic_tolerance": 0.02,
    "max_harmonic": 8,
    "max_peaks": 64,
}


def derive_noise_floor_baselines(
    freqs_hz: np.ndarray,
    ref_mean_db: np.ndarray,
    bands: Iterable[FrequencyBand],
    *,
    percentile: float = 10.0,
) -> dict[str, float]:
    """Derive per-band noise floor baselines from a reference mean curve."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ref_mean_db = np.asarray(ref_mean_db, dtype=np.float64)
    baselines: dict[str, float] = {}
    for band in bands:
        mask = (freqs_hz >= band.f_low) & (freqs_hz <= band.f_high)
        if not np.any(mask):
            baselines[band.name] = float(np.min(ref_mean_db))
            continue
        band_vals = ref_mean_db[mask]
        baselines[band.name] = float(np.percentile(band_vals, percentile))
    return baselines


def _local_peak_indices(
    mean_db: np.ndarray,
    *,
    neighborhood_bins: int,
    min_prominence_db: float,
) -> list[int]:
    indices: list[int] = []
    if mean_db.size < 3:
        return indices
    for idx in range(1, mean_db.size - 1):
        if mean_db[idx] <= mean_db[idx - 1] or mean_db[idx] < mean_db[idx + 1]:
            continue
        start = max(0, idx - neighborhood_bins)
        stop = min(mean_db.size, idx + neighborhood_bins + 1)
        neighborhood = np.concatenate((mean_db[start:idx], mean_db[idx + 1:stop]))
        if neighborhood.size == 0:
            continue
        baseline = float(np.median(neighborhood))
        prominence = float(mean_db[idx] - baseline)
        if prominence < min_prominence_db:
            continue
        indices.append(idx)
    return indices


def _assign_harmonic_groups(
    peaks: list[dict],
    *,
    tolerance: float,
    max_harmonic: int,
) -> None:
    peaks.sort(key=lambda p: p["frequency_hz"])
    assigned = [False] * len(peaks)
    group_id = 0
    for i, peak in enumerate(peaks):
        if assigned[i]:
            continue
        group_id += 1
        f0 = peak["frequency_hz"]
        peak["harmonic_group"] = group_id
        peak["harmonic_index"] = 1
        peak["fundamental_hz"] = f0
        assigned[i] = True
        for j in range(i + 1, len(peaks)):
            if assigned[j]:
                continue
            freq = peaks[j]["frequency_hz"]
            if f0 <= 0:
                continue
            ratio = freq / f0
            harmonic = int(round(ratio))
            if harmonic < 2 or harmonic > max_harmonic:
                continue
            if abs(ratio - harmonic) <= tolerance * harmonic:
                peaks[j]["harmonic_group"] = group_id
                peaks[j]["harmonic_index"] = harmonic
                peaks[j]["fundamental_hz"] = f0
                assigned[j] = True


def detect_tonal_peaks(
    freqs_hz: np.ndarray,
    mean_db: np.ndarray,
    *,
    bands: list[FrequencyBand],
    noise_floor_by_band: dict[str, float] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Detect tonal peaks in LTPSD mean curves with harmonic grouping."""
    cfg = {**DEFAULT_TONAL_DETECTION, **(config or {})}
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    mean_db = np.asarray(mean_db, dtype=np.float64)

    min_freq_hz = float(cfg["min_freq_hz"])
    max_freq_hz = cfg.get("max_freq_hz")
    max_freq_hz = float(max_freq_hz) if max_freq_hz is not None else None
    neighborhood_bins = int(cfg["neighborhood_bins"])
    min_prominence_db = float(cfg["min_prominence_db"])

    indices = _local_peak_indices(
        mean_db,
        neighborhood_bins=neighborhood_bins,
        min_prominence_db=min_prominence_db,
    )

    noise_floor_by_band = noise_floor_by_band or {}

    peaks: list[dict] = []
    for idx in indices:
        freq = float(freqs_hz[idx])
        if freq < min_freq_hz:
            continue
        if max_freq_hz is not None and freq > max_freq_hz:
            continue
        band = next((b for b in bands if b.f_low <= freq <= b.f_high), None)
        if band is None:
            continue
        level_db = float(mean_db[idx])
        noise_floor_db = noise_floor_by_band.get(band.name)
        delta_db = None if noise_floor_db is None else level_db - float(noise_floor_db)
        peaks.append(
            {
                "frequency_hz": freq,
                "level_db": level_db,
                "band_name": band.name,
                "noise_floor_db": noise_floor_db,
                "delta_db": delta_db,
            }
        )

    peaks = sorted(peaks, key=lambda p: p["level_db"], reverse=True)
    max_peaks = int(cfg.get("max_peaks", 0) or 0)
    if max_peaks > 0:
        peaks = peaks[:max_peaks]

    _assign_harmonic_groups(
        peaks,
        tolerance=float(cfg["harmonic_tolerance"]),
        max_harmonic=int(cfg["max_harmonic"]),
    )
    peaks.sort(key=lambda p: p["frequency_hz"])
    return peaks
