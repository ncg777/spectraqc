from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Status(str, Enum):
    """QC status enumeration."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class AudioBuffer:
    """Audio samples with metadata."""
    samples: np.ndarray  # float64, shape (n,) or (n, 2)
    fs: float
    duration: float
    channels: int
    backend: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrequencyBand:
    """Frequency band definition."""
    name: str
    f_low: float
    f_high: float


@dataclass(frozen=True)
class LongTermPSD:
    """Long-term PSD analysis result."""
    freqs: np.ndarray
    mean_db: np.ndarray
    var_db2: np.ndarray  # variance of dB across frames


@dataclass(frozen=True)
class BandMetrics:
    """Metrics computed for a single frequency band."""
    band: FrequencyBand
    mean_deviation_db: float
    max_deviation_db: float
    variance_ratio: float


@dataclass(frozen=True)
class GlobalMetrics:
    """Global audio metrics."""
    spectral_tilt_db_per_oct: float
    tilt_deviation_db_per_oct: float
    true_peak_dbtp: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    crest_factor_db: float | None = None
    lufs_i: float | None = None
    lra_lu: float | None = None
    tonal_peak_max_delta_db: float | None = None
    noise_floor_dbfs: float | None = None
    dynamic_range_db: float | None = None
    dynamic_range_lu: float | None = None


@dataclass(frozen=True)
class ThresholdResult:
    """Result of evaluating a metric against thresholds."""
    metric: str
    value: float
    units: str
    status: Status
    pass_limit: float
    warn_limit: float
    notes: str = ""


@dataclass(frozen=True)
class BandDecision:
    """Decision results for a frequency band."""
    band: FrequencyBand
    mean: ThresholdResult
    max: ThresholdResult
    variance: ThresholdResult


@dataclass(frozen=True)
class ProgramDecision:
    """Overall program decision with all band and global decisions."""
    overall_status: Status
    band_decisions: list[BandDecision]
    global_decisions: list[ThresholdResult]


@dataclass(frozen=True)
class ReferenceProfile:
    """Reference profile for QC comparison."""
    name: str
    kind: str
    version: str
    profile_hash_sha256: str
    analysis_lock_hash: str
    algorithm_ids: list[str]
    freqs_hz: np.ndarray
    ref_mean_db: np.ndarray
    bands: list[FrequencyBand]
    thresholds: dict
    analysis_lock: dict
    algorithm_registry: dict
    normalization: dict  # policy-driven: loudness + true peak
    noise_floor_by_band: dict[str, float] = field(default_factory=dict)
