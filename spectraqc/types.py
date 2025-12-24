from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    fs: float
    duration: float

@dataclass(frozen=True)
class FrequencyBand:
    name: str
    f_low: float
    f_high: float

@dataclass(frozen=True)
class BandMetrics:
    band: FrequencyBand
    mean_deviation_db: float
    max_deviation_db: float
    variance_ratio: float

@dataclass(frozen=True)
class GlobalMetrics:
    spectral_tilt_db_per_oct: float
    tilt_deviation_db_per_oct: float
    true_peak_dbtp: float | None = None
    lufs_i: float | None = None
