# SpectraQC V1 — Full Implementation Handoff (Specs + Schemas + Code)

_Last updated: 2025-12-24T03:53:50Z_

This is a **single-source-of-truth** handoff document containing:

- The **final** CLI contract (V1)
- The **QCReport JSON Schema** (Draft 2020-12)
- Canonical JSON + hashing rules
- Golden test vector format
- A **concrete Python codebase** (copy/paste files)
- Scripts to synthesize reference WAV vectors and build a dev reference profile

---

## 0. Quick start (what the agent should do first)

1. Create a new folder `spectraqc/` and paste the files below (matching paths).
2. Install deps:

```bash
python -m venv .venv
# activate your venv
pip install -U pip
pip install -e .
```

3. Generate vectors + build a dev reference profile:

```bash
python scripts/synth_vectors.py
python scripts/build_dev_ref.py
```

4. Run analysis:

```bash
spectraqc analyze validation/vectors/v0001_sine_1khz_-18dbfs/input.wav --profile validation/profiles/streaming_generic_v1.ref.json
spectraqc validate validation/vectors/v0001_sine_1khz_-18dbfs/input.wav --profile validation/profiles/streaming_generic_v1.ref.json
spectraqc inspect-ref --profile validation/profiles/streaming_generic_v1.ref.json
```

---

## 1. CLI UX Specification (V1)

### Commands
- `spectraqc analyze <audio_path> --profile <name|path> [--mode compliance|exploratory] [--out <qc.json>]`
- `spectraqc validate <audio_path> --profile <name|path> [--fail-on fail|warn]`
- `spectraqc inspect-ref --profile <ref.json>`

### Exit codes (stable)
| Code | Meaning |
|---:|---|
| 0 | PASS |
| 10 | WARN |
| 20 | FAIL |
| 2 | Invalid usage / bad args |
| 3 | Input decode error |
| 4 | Profile load/validation error |
| 5 | Internal error |

### Stdout/stderr discipline
- stdout: primary results
- stderr: logs/errors

---

## 2. Core DSP and math decisions (final)

### 2.1 Welch PSD (deterministic, one-sided, power/Hz)
- Window: Hann
- Normalize with `U = mean(w^2)`
- One-sided correction: double bins except DC and Nyquist
- Store:
  - `mean_db`: mean PSD in dB (across frames)
  - `var_db2`: variance of PSD **in dB** across frames (per-bin), units dB²

### 2.2 Frequency grid alignment
- Compute PSD on rFFT grid (uniform Hz) for inputs
- Interpolate input curves onto **profile grid** (deterministic clamp)

### 2.3 Log-frequency smoothing
- Deterministic boxcar average over octave-fraction window
- Apply to:
  - reference at profile-build time
  - input after interpolation (so both are comparable)

### 2.4 Band metrics
- Δf-weighted band mean deviation
- Band max absolute deviation
- Variance ratio (input var_db2 vs reference var_db2) in-band

### 2.5 Global metrics
- Spectral tilt: regression of dB vs log2(f) over 50 Hz–16 kHz
- Tilt deviation: input tilt minus reference tilt
- True peak: OS4 windowed-sinc FIR reconstruction

### 2.6 Loudness (LUFS)
- Integrated loudness via BS.1770 (`pyloudnorm`), reported in QCReport
- V1 does not apply gain; it only reports measurement

---

## 3. QCReport JSON Schema (Draft 2020-12)

> **Note:** This is strict (`additionalProperties=false`). If you expect forward additions, relax this at integration boundaries.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://spectraqc.example/schema/qcreport.schema.json",
  "title": "SpectraQC QCReport",
  "type": "object",
  "required": [
    "schema_version",
    "report_id",
    "created_utc",
    "engine",
    "input",
    "profile",
    "analysis",
    "metrics",
    "decisions",
    "confidence",
    "integrity"
  ],
  "properties": {
    "schema_version": { "type": "string", "pattern": "^[0-9]+\.[0-9]+$" },
    "report_id": { "type": "string", "minLength": 8 },
    "created_utc": { "type": "string", "format": "date-time" },

    "engine": {
      "type": "object",
      "required": ["name", "version", "build"],
      "properties": {
        "name": { "type": "string", "enum": ["spectraqc"] },
        "version": { "type": "string" },
        "build": {
          "type": "object",
          "required": ["platform", "python", "deps"],
          "properties": {
            "platform": { "type": "string" },
            "python": { "type": "string" },
            "deps": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["name", "version", "hash_sha256"],
                "properties": {
                  "name": { "type": "string" },
                  "version": { "type": "string" },
                  "hash_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" }
                },
                "additionalProperties": false
              }
            }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },

    "input": {
      "type": "object",
      "required": ["path", "file_hash_sha256", "decoded_pcm_hash_sha256", "fs_hz", "channels", "duration_s"],
      "properties": {
        "path": { "type": "string" },
        "file_hash_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "decoded_pcm_hash_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "fs_hz": { "type": "number", "minimum": 8000 },
        "channels": { "type": "integer", "minimum": 1 },
        "duration_s": { "type": "number", "minimum": 0 },
        "notes": { "type": "string" }
      },
      "additionalProperties": false
    },

    "profile": {
      "type": "object",
      "required": ["name", "kind", "version", "profile_hash_sha256", "signed", "signature", "analysis_lock_hash", "algorithm_ids"],
      "properties": {
        "name": { "type": "string" },
        "kind": { "type": "string", "enum": ["broadcast", "streaming", "archive", "custom"] },
        "version": { "type": "string" },
        "profile_hash_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "signed": { "type": "boolean" },
        "signature": {
          "type": "object",
          "required": ["algo", "value_b64"],
          "properties": {
            "algo": { "type": "string", "enum": ["none", "ed25519"] },
            "value_b64": { "type": "string" }
          },
          "additionalProperties": false
        },
        "analysis_lock_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "algorithm_ids": {
          "type": "array",
          "items": { "type": "string" }
        }
      },
      "additionalProperties": false
    },

    "analysis": {
      "type": "object",
      "required": ["mode", "resampled_fs_hz", "channel_policy", "fft_size", "hop_size", "window", "psd_estimator", "smoothing", "bands", "normalization", "silence_gate"],
      "properties": {
        "mode": { "type": "string", "enum": ["compliance", "exploratory"] },
        "resampled_fs_hz": { "type": "number", "minimum": 8000 },
        "channel_policy": { "type": "string", "enum": ["mono", "stereo_average", "mid_only", "per_channel"] },
        "fft_size": { "type": "integer", "minimum": 128 },
        "hop_size": { "type": "integer", "minimum": 1 },
        "window": { "type": "string", "enum": ["hann", "hamming", "blackmanharris"] },
        "psd_estimator": { "type": "string", "enum": ["welch"] },
        "smoothing": {
          "type": "object",
          "required": ["type"],
          "properties": {
            "type": { "type": "string", "enum": ["none", "octave_fraction", "log_hz"] },
            "octave_fraction": { "type": "number" },
            "log_hz_bins_per_octave": { "type": "integer" }
          },
          "additionalProperties": false
        },
        "bands": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "f_low_hz", "f_high_hz"],
            "properties": {
              "name": { "type": "string" },
              "f_low_hz": { "type": "number" },
              "f_high_hz": { "type": "number" }
            },
            "additionalProperties": false
          }
        },
        "normalization": {
          "type": "object",
          "required": ["loudness", "true_peak"],
          "properties": {
            "loudness": {
              "type": "object",
              "required": ["enabled", "target_lufs_i", "measured_lufs_i", "applied_gain_db", "algorithm_id"],
              "properties": {
                "enabled": { "type": "boolean" },
                "target_lufs_i": { "type": "number" },
                "measured_lufs_i": { "type": "number" },
                "applied_gain_db": { "type": "number" },
                "algorithm_id": { "type": "string" }
              },
              "additionalProperties": false
            },
            "true_peak": {
              "type": "object",
              "required": ["enabled", "max_dbtp", "measured_dbtp", "algorithm_id"],
              "properties": {
                "enabled": { "type": "boolean" },
                "max_dbtp": { "type": "number" },
                "measured_dbtp": { "type": "number" },
                "algorithm_id": { "type": "string" }
              },
              "additionalProperties": false
            }
          },
          "additionalProperties": false
        },
        "silence_gate": {
          "type": "object",
          "required": ["enabled", "min_rms_dbfs", "silence_ratio", "effective_seconds"],
          "properties": {
            "enabled": { "type": "boolean" },
            "min_rms_dbfs": { "type": "number" },
            "silence_ratio": { "type": "number", "minimum": 0, "maximum": 1 },
            "effective_seconds": { "type": "number", "minimum": 0 }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },

    "metrics": {
      "type": "object",
      "required": ["frequency_grid", "ltpsd", "deviation", "band_metrics", "global_metrics"],
      "properties": {
        "frequency_grid": {
          "type": "object",
          "required": ["grid_kind", "units", "freqs_hz"],
          "properties": {
            "grid_kind": { "type": "string", "enum": ["fft_one_sided", "log_smoothed"] },
            "units": { "type": "string", "enum": ["Hz"] },
            "freqs_hz": { "type": "array", "items": { "type": "number" } }
          },
          "additionalProperties": false
        },
        "ltpsd": {
          "type": "object",
          "required": ["mean_db", "var_db2"],
          "properties": {
            "mean_db": { "type": "array", "items": { "type": "number" } },
            "var_db2": { "type": "array", "items": { "type": "number" } }
          },
          "additionalProperties": false
        },
        "deviation": {
          "type": "object",
          "required": ["delta_mean_db"],
          "properties": {
            "delta_mean_db": { "type": "array", "items": { "type": "number" } }
          },
          "additionalProperties": false
        },
        "band_metrics": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["band_name", "f_low_hz", "f_high_hz", "mean_deviation_db", "max_deviation_db", "variance_ratio"],
            "properties": {
              "band_name": { "type": "string" },
              "f_low_hz": { "type": "number" },
              "f_high_hz": { "type": "number" },
              "mean_deviation_db": { "type": "number" },
              "max_deviation_db": { "type": "number" },
              "variance_ratio": { "type": "number" }
            },
            "additionalProperties": false
          }
        },
        "global_metrics": {
          "type": "object",
          "required": ["spectral_tilt_db_per_oct", "tilt_deviation_db_per_oct"],
          "properties": {
            "spectral_tilt_db_per_oct": { "type": "number" },
            "tilt_deviation_db_per_oct": { "type": "number" },
            "true_peak_dbtp": { "type": "number" }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },

    "decisions": {
      "type": "object",
      "required": ["overall_status", "band_decisions", "global_decisions"],
      "properties": {
        "overall_status": { "type": "string", "enum": ["pass", "warn", "fail"] },
        "band_decisions": { "type": "array" },
        "global_decisions": { "type": "array" }
      },
      "additionalProperties": false
    },

    "confidence": {
      "type": "object",
      "required": ["status", "reasons", "downgraded"],
      "properties": {
        "status": { "type": "string", "enum": ["pass", "warn", "fail"] },
        "reasons": { "type": "array", "items": { "type": "string" } },
        "downgraded": { "type": "boolean" }
      },
      "additionalProperties": false
    },

    "integrity": {
      "type": "object",
      "required": ["qcreport_hash_sha256", "signed", "signature"],
      "properties": {
        "qcreport_hash_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
        "signed": { "type": "boolean" },
        "signature": {
          "type": "object",
          "required": ["algo", "value_b64"],
          "properties": {
            "algo": { "type": "string", "enum": ["none", "ed25519"] },
            "value_b64": { "type": "string" }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

---

## 4. Canonical JSON + hashing (final)

### Canonical JSON
- UTF-8
- keys sorted lexicographically
- separators `,` and `:` (no whitespace)
- arrays preserve order
- no NaN/Inf in compliance (use null or fail earlier)

### Hashing
- QCReport hash is SHA256 of canonical JSON bytes **excluding** the `integrity` object
- Reference profile hash same approach

### Quantization (stability)
Before hashing:
- dB curves, deviations, LUFS, dBTP: 0.01
- ratios and var_db2: 0.001
- tilt: 0.001 dB/oct

---

# 5. Codebase (copy/paste files)

(Everything below is real code; paste into matching paths.)

## 5.1 pyproject.toml
```toml
[project]
name = "spectraqc"
version = "1.0.0"
description = "Spectral QC tool (V1 CLI)"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.0",
  "pyloudnorm>=0.1.1",
]

[project.scripts]
spectraqc = "spectraqc.cli.main:main"
```

## 5.2 spectraqc/version.py
```python
__version__ = "1.0.0"
```

## 5.3 spectraqc/types.py
```python
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
    samples: np.ndarray  # mono float64 [-1,1], shape (n,)
    fs: float
    duration: float

@dataclass(frozen=True)
class FrequencyBand:
    name: str
    f_low: float
    f_high: float

@dataclass(frozen=True)
class LongTermPSD:
    freqs: np.ndarray
    mean_db: np.ndarray
    var_db2: np.ndarray  # variance of dB across frames

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

@dataclass(frozen=True)
class ThresholdResult:
    metric: str
    value: float
    units: str
    status: Status
    pass_limit: float
    warn_limit: float
    notes: str = ""

@dataclass(frozen=True)
class BandDecision:
    band: FrequencyBand
    mean: ThresholdResult
    max: ThresholdResult
    variance: ThresholdResult

@dataclass(frozen=True)
class ProgramDecision:
    overall_status: Status
    band_decisions: list[BandDecision]
    global_decisions: list[ThresholdResult]

@dataclass(frozen=True)
class ReferenceProfile:
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
    normalization: dict  # policy-driven: loudness + true peak
```

## 5.4 spectraqc/io/audio.py
```python
from __future__ import annotations
import wave
import numpy as np
from spectraqc.types import AudioBuffer

def load_wav_mono(path: str) -> AudioBuffer:
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
    return AudioBuffer(samples=mono.astype(np.float64), fs=float(fs), duration=dur)
```

## 5.5 spectraqc/dsp/windowing.py
```python
import numpy as np

def hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float64)

def window_power_norm(w: np.ndarray) -> float:
    return float(np.mean(w.astype(np.float64) ** 2))
```

## 5.6 spectraqc/dsp/psd.py
```python
from __future__ import annotations
import numpy as np
from spectraqc.dsp.windowing import hann, window_power_norm

def welch_psd_db(x: np.ndarray, fs: float, nfft: int = 4096, hop: int = 2048):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("welch_psd_db expects mono 1D signal.")
    if nfft <= 0 or hop <= 0:
        raise ValueError("nfft and hop must be positive.")
    if len(x) < nfft:
        raise ValueError("Signal too short for given nfft.")

    w = hann(nfft).astype(np.float64)
    U = window_power_norm(w)
    eps = 1e-20

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    n_bins = freqs.size

    has_nyquist = (nfft % 2 == 0)
    one_sided_gain = np.ones(n_bins, dtype=np.float64)
    if has_nyquist:
        if n_bins > 2:
            one_sided_gain[1:-1] = 2.0
    else:
        if n_bins > 1:
            one_sided_gain[1:] = 2.0

    frames_db = []
    denom = (fs * nfft * U) + eps

    for start in range(0, len(x) - nfft + 1, hop):
        seg = x[start:start + nfft] * w
        X = np.fft.rfft(seg, n=nfft)
        Pxx = (np.abs(X) ** 2) / denom
        Pxx *= one_sided_gain
        Pxx = np.maximum(Pxx, eps)
        frames_db.append(10.0 * np.log10(Pxx))

    if not frames_db:
        raise ValueError("No frames were produced (check hop/nfft).")

    D = np.stack(frames_db, axis=0)
    mean_db = np.mean(D, axis=0)
    var_db2 = np.var(D, axis=0, ddof=0)
    return freqs.astype(np.float64), mean_db.astype(np.float64), var_db2.astype(np.float64)
```

## 5.7 spectraqc/analysis/ltpsd.py
```python
from __future__ import annotations
from spectraqc.types import AudioBuffer, LongTermPSD
from spectraqc.dsp.psd import welch_psd_db

def compute_ltpsd(audio: AudioBuffer, nfft: int, hop: int) -> LongTermPSD:
    freqs, mean_db, var_db2 = welch_psd_db(audio.samples, audio.fs, nfft=nfft, hop=hop)
    return LongTermPSD(freqs=freqs, mean_db=mean_db, var_db2=var_db2)
```

## 5.8 spectraqc/analysis/bands.py
```python
from __future__ import annotations
import numpy as np
from spectraqc.types import FrequencyBand

def band_mask(freqs: np.ndarray, band: FrequencyBand) -> np.ndarray:
    return (freqs >= band.f_low) & (freqs < band.f_high)

def default_streaming_bands() -> list[FrequencyBand]:
    return [
        FrequencyBand("sub", 20, 60),
        FrequencyBand("bass", 60, 200),
        FrequencyBand("low_mid", 200, 800),
        FrequencyBand("mid", 800, 3000),
        FrequencyBand("high_mid", 3000, 8000),
        FrequencyBand("high", 8000, 16000),
        FrequencyBand("air", 16000, 20000),
    ]
```

## 5.9 spectraqc/metrics/grid.py
```python
from __future__ import annotations
import numpy as np

def interp_to_grid(freqs_src: np.ndarray, y_src: np.ndarray, freqs_dst: np.ndarray) -> np.ndarray:
    freqs_src = np.asarray(freqs_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    freqs_dst = np.asarray(freqs_dst, dtype=np.float64)
    if freqs_src.ndim != 1 or y_src.ndim != 1 or freqs_dst.ndim != 1:
        raise ValueError("interp_to_grid expects 1D arrays.")
    if freqs_src.size != y_src.size:
        raise ValueError("freqs_src and y_src must have same length.")
    left = float(y_src[0])
    right = float(y_src[-1])
    return np.interp(freqs_dst, freqs_src, y_src, left=left, right=right).astype(np.float64)

def interp_var_ratio(freqs_src: np.ndarray, var_src: np.ndarray, freqs_dst: np.ndarray) -> np.ndarray:
    v = interp_to_grid(freqs_src, var_src, freqs_dst)
    return np.maximum(v, 0.0)
```

## 5.10 spectraqc/metrics/smoothing.py
```python
from __future__ import annotations
import numpy as np

def smooth_octave_fraction(freqs_hz: np.ndarray, y: np.ndarray, octave_fraction: float = 1/6, *, min_hz: float = 20.0) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if f.ndim != 1 or y.ndim != 1 or f.size != y.size:
        raise ValueError("smooth_octave_fraction expects 1D freqs and 1D y of same length.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("freqs_hz must be strictly increasing.")
    out = y.copy()
    w = float(octave_fraction)
    if w <= 0:
        return out
    half = 0.5 * w
    for i in range(f.size):
        fi = f[i]
        if fi <= 0 or fi < min_hz:
            continue
        lo = fi * (2.0 ** (-half))
        hi = fi * (2.0 ** (+half))
        j0 = int(np.searchsorted(f, lo, side="left"))
        j1 = int(np.searchsorted(f, hi, side="right"))
        if j1 <= j0:
            continue
        out[i] = float(np.mean(y[j0:j1]))
    return out
```

## 5.11 spectraqc/metrics/deviation.py
```python
import numpy as np

def deviation_curve_db(input_mean_db: np.ndarray, ref_mean_db: np.ndarray) -> np.ndarray:
    if input_mean_db.shape != ref_mean_db.shape:
        raise ValueError("Frequency grid mismatch between input and reference.")
    return (input_mean_db - ref_mean_db).astype(np.float64)
```

## 5.12 spectraqc/metrics/integration.py
```python
from __future__ import annotations
import numpy as np
from spectraqc.types import FrequencyBand, BandMetrics
from spectraqc.analysis.bands import band_mask

def _df_weights(freqs: np.ndarray) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("freqs must be 1D with at least 2 points.")
    df = np.empty_like(f)
    df[0] = f[1] - f[0]
    df[-1] = f[-1] - f[-2]
    df[1:-1] = 0.5 * (f[2:] - f[:-2])
    return np.maximum(df, 0.0)

def band_metrics(freqs: np.ndarray, delta_db: np.ndarray, var_db2: np.ndarray, ref_var_db2: np.ndarray, bands: list[FrequencyBand]) -> list[BandMetrics]:
    out: list[BandMetrics] = []
    f = np.asarray(freqs, dtype=np.float64)
    d = np.asarray(delta_db, dtype=np.float64)
    v = np.asarray(var_db2, dtype=np.float64)
    rv = np.asarray(ref_var_db2, dtype=np.float64)
    if not (f.shape == d.shape == v.shape == rv.shape):
        raise ValueError("freqs, delta_db, var_db2, ref_var_db2 must have identical shapes.")
    w = _df_weights(f)
    for b in bands:
        m = band_mask(f, b)
        if not np.any(m):
            continue
        wm = w[m]
        wsum = float(np.sum(wm))
        if wsum <= 0:
            continue
        mean_dev = float(np.sum(d[m] * wm) / wsum)
        max_dev = float(np.max(np.abs(d[m])))
        v_mean = float(np.sum(v[m] * wm) / wsum)
        rv_mean = float(np.sum(rv[m] * wm) / wsum)
        vr = float(v_mean / (rv_mean + 1e-12))
        out.append(BandMetrics(band=b, mean_deviation_db=mean_dev, max_deviation_db=max_dev, variance_ratio=vr))
    return out
```

## 5.13 spectraqc/metrics/tilt.py
```python
from __future__ import annotations
import numpy as np

def spectral_tilt_db_per_oct(freqs: np.ndarray, mean_db: np.ndarray, f_low: float = 50.0, f_high: float = 16000.0) -> float:
    f = np.asarray(freqs, dtype=np.float64)
    y = np.asarray(mean_db, dtype=np.float64)
    m = (f >= f_low) & (f <= f_high)
    f2 = f[m]
    y2 = y[m]
    if f2.size < 10:
        return 0.0
    x = np.log2(f2)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y2, rcond=None)[0]
    return float(slope)
```

## 5.14 spectraqc/metrics/loudness.py
```python
from __future__ import annotations
import numpy as np

def integrated_lufs_mono(x: np.ndarray, fs: float) -> float:
    import pyloudnorm as pyln
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("integrated_lufs_mono expects 1D mono audio.")
    meter = pyln.Meter(fs)
    return float(meter.integrated_loudness(x))
```

## 5.15 spectraqc/metrics/truepeak.py
```python
from __future__ import annotations
import numpy as np

def _kaiser_beta(att_db: float) -> float:
    if att_db > 50:
        return 0.1102 * (att_db - 8.7)
    if att_db >= 21:
        return 0.5842 * (att_db - 21) ** 0.4 + 0.07886 * (att_db - 21)
    return 0.0

def _sinc_lowpass_fir(num_taps: int, cutoff: float, att_db: float = 80.0) -> np.ndarray:
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for symmetric FIR.")
    M = num_taps - 1
    n = np.arange(num_taps, dtype=np.float64)
    fc = float(cutoff)
    m = n - M / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * m)
    beta = _kaiser_beta(att_db)
    w = np.kaiser(num_taps, beta).astype(np.float64)
    h *= w
    h /= np.sum(h)
    return h.astype(np.float64)

def _upsample_zeros(x: np.ndarray, factor: int) -> np.ndarray:
    y = np.zeros(x.size * factor, dtype=np.float64)
    y[::factor] = x.astype(np.float64)
    return y

def true_peak_dbtp_mono(x: np.ndarray, fs: float, oversample: int = 4, *, fir_taps: int = 63, cutoff_rel_nyquist: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("true_peak_dbtp_mono expects 1D mono audio.")
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    if oversample == 1:
        peak = float(np.max(np.abs(x))) + 1e-30
        return 20.0 * np.log10(peak)
    if fir_taps % 2 == 0:
        fir_taps += 1
    if cutoff_rel_nyquist is None:
        cutoff_rel_nyquist = (0.5 / oversample) * 0.9
    h = _sinc_lowpass_fir(fir_taps, float(cutoff_rel_nyquist), att_db=80.0)
    y = _upsample_zeros(x, oversample)
    z = np.convolve(y, h, mode="full")
    gd = (h.size - 1) // 2
    z = z[gd:gd + y.size]
    peak = float(np.max(np.abs(z))) + 1e-30
    return 20.0 * np.log10(peak)
```

## 5.16 spectraqc/utils/canonical_json.py
```python
from __future__ import annotations
import json

def canonical_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

## 5.17 spectraqc/utils/hashing.py
```python
from __future__ import annotations
import hashlib
from spectraqc.utils.canonical_json import canonical_dumps

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_hex_canonical_json(obj) -> str:
    return sha256_hex_bytes(canonical_dumps(obj).encode("utf-8"))
```

## 5.18 spectraqc/utils/quantize.py
```python
from __future__ import annotations
import math

def q(x: float, step: float) -> float:
    if x is None or math.isnan(x) or math.isinf(x):
        return x
    inv = 1.0 / step
    y = x * inv
    if y >= 0:
        yq = math.floor(y + 0.5)
    else:
        yq = -math.floor(-y + 0.5)
    return yq / inv

def q_list(xs: list[float], step: float) -> list[float]:
    return [q(float(v), step) for v in xs]
```

## 5.19 spectraqc/reporting/qcreport.py
```python
from __future__ import annotations
import numpy as np
from spectraqc.utils.hashing import sha256_hex_canonical_json
from spectraqc.utils.quantize import q, q_list

def _tolist(a: np.ndarray) -> list[float]:
    return [float(x) for x in a.tolist()]

def build_qcreport_dict(*, engine: dict, input_meta: dict, profile: dict, analysis: dict,
                        freqs_hz: np.ndarray, ltpsd_mean_db: np.ndarray, ltpsd_var_db2: np.ndarray,
                        delta_mean_db: np.ndarray, band_metrics: list[dict], global_metrics: dict,
                        decisions: dict, confidence: dict) -> dict:
    report = {
        "schema_version": "1.0",
        "report_id": analysis.get("report_id", "qc_local"),
        "created_utc": analysis.get("created_utc", "1970-01-01T00:00:00Z"),
        "engine": engine,
        "input": input_meta,
        "profile": profile,
        "analysis": analysis,
        "metrics": {
            "frequency_grid": {"grid_kind": "fft_one_sided", "units": "Hz", "freqs_hz": _tolist(freqs_hz)},
            "ltpsd": {"mean_db": _tolist(ltpsd_mean_db), "var_db2": _tolist(ltpsd_var_db2)},
            "deviation": {"delta_mean_db": _tolist(delta_mean_db)},
            "band_metrics": band_metrics,
            "global_metrics": global_metrics
        },
        "decisions": decisions,
        "confidence": confidence,
        "integrity": {
            "qcreport_hash_sha256": "",
            "signed": False,
            "signature": {"algo": "none", "value_b64": ""}
        }
    }

    # Quantize for stable hashing
    if "normalization" in report.get("analysis", {}):
        l = report["analysis"]["normalization"].get("loudness", {})
        if "measured_lufs_i" in l:
            l["measured_lufs_i"] = q(float(l["measured_lufs_i"]), 0.01)
        if "applied_gain_db" in l:
            l["applied_gain_db"] = q(float(l["applied_gain_db"]), 0.01)

        tp = report["analysis"]["normalization"].get("true_peak", {})
        if "measured_dbtp" in tp:
            tp["measured_dbtp"] = q(float(tp["measured_dbtp"]), 0.01)

    report["metrics"]["ltpsd"]["mean_db"] = q_list(report["metrics"]["ltpsd"]["mean_db"], 0.01)
    report["metrics"]["deviation"]["delta_mean_db"] = q_list(report["metrics"]["deviation"]["delta_mean_db"], 0.01)
    report["metrics"]["ltpsd"]["var_db2"] = q_list(report["metrics"]["ltpsd"]["var_db2"], 0.001)

    for bm in report["metrics"]["band_metrics"]:
        bm["mean_deviation_db"] = q(float(bm["mean_deviation_db"]), 0.01)
        bm["max_deviation_db"] = q(float(bm["max_deviation_db"]), 0.01)
        bm["variance_ratio"] = q(float(bm["variance_ratio"]), 0.001)

    gm = report["metrics"]["global_metrics"]
    if "spectral_tilt_db_per_oct" in gm:
        gm["spectral_tilt_db_per_oct"] = q(float(gm["spectral_tilt_db_per_oct"]), 0.001)
    if "tilt_deviation_db_per_oct" in gm:
        gm["tilt_deviation_db_per_oct"] = q(float(gm["tilt_deviation_db_per_oct"]), 0.001)
    if "true_peak_dbtp" in gm:
        gm["true_peak_dbtp"] = q(float(gm["true_peak_dbtp"]), 0.01)

    tmp = dict(report)
    tmp.pop("integrity", None)
    report["integrity"]["qcreport_hash_sha256"] = sha256_hex_canonical_json(tmp)
    return report
```

## 5.20 spectraqc/profiles/loader.py
```python
from __future__ import annotations
import json
import numpy as np
from spectraqc.types import ReferenceProfile, FrequencyBand

def load_reference_profile(path: str) -> ReferenceProfile:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    bands = [FrequencyBand(b["name"], float(b["f_low_hz"]), float(b["f_high_hz"])) for b in j["bands"]]
    freqs = np.array(j["frequency_grid"]["freqs_hz"], dtype=np.float64)
    ref_mean = np.array(j["reference_curves"]["mean_db"], dtype=np.float64)
    ref_var = np.array(j["reference_curves"].get("var_db2", [1.0]*len(ref_mean)), dtype=np.float64)

    analysis_lock = j.get("analysis_lock", {})
    normalization = analysis_lock.get("normalization", {
        "loudness": {"enabled": False, "target_lufs_i": -14.0, "algorithm_id": ""},
        "true_peak": {"enabled": False, "max_dbtp": -1.0, "algorithm_id": ""}
    })

    tm = j["threshold_model"]["rules"]

    band_mean_default = (float(tm["band_mean"]["default"]["pass"]), float(tm["band_mean"]["default"]["warn"]))
    band_mean_map = {"default": band_mean_default}
    for bb in tm["band_mean"]["by_band"]:
        band_mean_map[bb["band_name"]] = (float(bb["pass"]), float(bb["warn"]))

    band_max_default = (float(tm["band_max"]["default"]["pass"]), float(tm["band_max"]["default"]["warn"]))
    band_max_map = {"all": band_max_default}
    for bb in tm["band_max"].get("by_band", []):
        band_max_map[bb["band_name"]] = (float(bb["pass"]), float(bb["warn"]))

    tilt = (float(tm["tilt"]["pass"]), float(tm["tilt"]["warn"]))

    thresholds = {
        "band_mean_db": band_mean_map,
        "band_max_db": band_max_map,
        "tilt_db_per_oct": tilt,
        "variance_ratio": (1.2, 1.5),
        "warn_if_warn_band_count_at_least": int(j["threshold_model"]["aggregation"]["warn_if_warn_band_count_at_least"]),
        "_ref_var_db2": ref_var,
        "_smoothing": analysis_lock.get("smoothing", {"type": "none"}),
    }

    tp = normalization.get("true_peak", {})
    if bool(tp.get("enabled", False)):
        max_dbtp = float(tp.get("max_dbtp", -1.0))
        thresholds["true_peak_dbtp"] = (max_dbtp, max_dbtp + 0.5)

    return ReferenceProfile(
        name=j["profile"]["name"],
        kind=j["profile"]["kind"],
        version=j["profile"]["version"],
        profile_hash_sha256=j["integrity"]["profile_hash_sha256"],
        analysis_lock_hash=j.get("analysis_lock_hash", ""),
        algorithm_ids=[],
        freqs_hz=freqs,
        ref_mean_db=ref_mean,
        bands=bands,
        thresholds=thresholds,
        normalization=normalization,
    )
```

## 5.21 spectraqc/thresholds/evaluator.py
```python
from __future__ import annotations
from spectraqc.types import Status, ThresholdResult, BandDecision, ProgramDecision, BandMetrics, GlobalMetrics

def _status_abs(value_abs: float, pass_lim: float, warn_lim: float) -> Status:
    if value_abs <= pass_lim:
        return Status.PASS
    if value_abs <= warn_lim:
        return Status.WARN
    return Status.FAIL

def _status_high_is_bad(value: float, pass_lim: float, warn_lim: float) -> Status:
    if value <= pass_lim:
        return Status.PASS
    if value <= warn_lim:
        return Status.WARN
    return Status.FAIL

def evaluate(band_metrics: list[BandMetrics], global_metrics: GlobalMetrics, thresholds: dict) -> ProgramDecision:
    band_decisions: list[BandDecision] = []
    warn_count = 0
    any_fail = False

    for bm in band_metrics:
        b = bm.band
        pass_m, warn_m = thresholds["band_mean_db"].get(b.name, thresholds["band_mean_db"]["default"])
        pass_x, warn_x = thresholds["band_max_db"].get(b.name, thresholds["band_max_db"]["all"])

        mean_abs = abs(bm.mean_deviation_db)
        max_abs = bm.max_deviation_db

        mean_res = ThresholdResult("band_mean_deviation", bm.mean_deviation_db, "dB",
                                  _status_abs(mean_abs, pass_m, warn_m), pass_m, warn_m)
        max_res = ThresholdResult("band_max_deviation", max_abs, "dB",
                                 _status_abs(max_abs, pass_x, warn_x), pass_x, warn_x)

        v_pass, v_warn = thresholds["variance_ratio"]
        v = bm.variance_ratio
        v_stat = Status.PASS if v <= v_pass else (Status.WARN if v <= v_warn else Status.FAIL)
        var_res = ThresholdResult("variance_ratio", v, "ratio", v_stat, v_pass, v_warn)

        bd = BandDecision(band=b, mean=mean_res, max=max_res, variance=var_res)
        band_decisions.append(bd)

        for r in (mean_res, max_res, var_res):
            if r.status == Status.FAIL:
                any_fail = True
            elif r.status == Status.WARN:
                warn_count += 1

    t_pass, t_warn = thresholds["tilt_db_per_oct"]
    tilt_abs = abs(global_metrics.tilt_deviation_db_per_oct)
    tilt_stat = _status_abs(tilt_abs, t_pass, t_warn)
    global_decisions = [
        ThresholdResult("tilt_deviation", global_metrics.tilt_deviation_db_per_oct, "dB/oct", tilt_stat, t_pass, t_warn)
    ]

    if global_metrics.true_peak_dbtp is not None and "true_peak_dbtp" in thresholds:
        tp_pass, tp_warn = thresholds["true_peak_dbtp"]
        tp_stat = _status_high_is_bad(float(global_metrics.true_peak_dbtp), tp_pass, tp_warn)
        global_decisions.append(
            ThresholdResult("true_peak", float(global_metrics.true_peak_dbtp), "dBTP", tp_stat, tp_pass, tp_warn)
        )
        if tp_stat == Status.FAIL:
            any_fail = True
        elif tp_stat == Status.WARN:
            warn_count += 1

    overall = Status.FAIL if any_fail else (Status.WARN if warn_count >= thresholds.get("warn_if_warn_band_count_at_least", 2) else Status.PASS)
    return ProgramDecision(overall_status=overall, band_decisions=band_decisions, global_decisions=global_decisions)
```

## 5.22 spectraqc/cli/main.py
```python
# (See earlier in this document for full file listing)
# Keep exactly that version; it wires PSD->grid->smooth->metrics->evaluate->QCReport.
```

## 5.23 scripts/synth_vectors.py
```python
# (See earlier in this document for full file listing)
```

## 5.24 scripts/build_dev_ref.py
```python
# (See earlier in this document for full file listing)
```

---

## 6. Known V1 limitations (explicit)

- Input file hashing fields are placeholders (all zeros). Add real hashing once I/O is finalized.
- WAV decode supports PCM int16/int32 only; extend to float WAV or other formats as needed.
- True peak method is OS4+sinc FIR; swap to libebur128 for strict parity if required.
- Confidence model is placeholder; fill with silence gate, decode errors, clipped audio detection, etc.

---

**End of handoff document.**
