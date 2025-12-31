# SpectraQC Documentation

## Purpose and usefulness

SpectraQC is a deterministic spectral quality control tool for audio. Its purpose is to let you verify that an audio file or an entire corpus conforms to a declared reference profile with reproducible, audit-friendly outputs. It is useful when you need technical consistency checks that can be rerun later with the same inputs and get the same decisions, without subjective judgments or automatic correction.

Key goals:
- Determinism: same inputs produce the same metrics and decisions within floating-point tolerance.
- Auditability: outputs are structured, hashed, and can be archived.
- Explainability: decisions include per-band and global metric details.
- Corpus-scale operation: batch analysis and aggregate reporting.

## Scope and version alignment

This repository implements a V1 CLI with a V2 reference profile schema. The V1 CLI is described in `docs/SpectraQC_V1_Full_Implementation.md`, and the V2 scope is defined in `docs/V2_SCOPE.md`. The profile schema is `docs/reference_profile_v2.schema.json`. The corpus manifest schema is `docs/corpus_manifest_v1.schema.json`.

The documentation below is authoritative for the behavior of the current codebase. When a document conflicts with code, code is normative.

## Formal specification

### Terminology
- Audio buffer: decoded samples in float64 and associated metadata (sample rate, duration, channels, warnings).
- Reference profile: a `.ref.json` profile that defines a frequency grid, reference curves, bands, and thresholds.
- QCReport: the JSON report produced by analysis.
- Decision: PASS, WARN, or FAIL for the full program and for individual metrics.

### Determinism and reproducibility
The system MUST be deterministic for the same inputs, reference profile, and runtime environment. To support reproducibility, it also:
- Uses fixed algorithms (Welch PSD, linear interpolation, log-frequency smoothing).
- Quantizes numeric outputs prior to hashing.
- Computes hashes from canonical JSON (sorted keys, no whitespace) that exclude the integrity block itself.

### Inputs and outputs
- Inputs:
  - Audio file path (any supported format).
  - Reference profile `.ref.json` (schema v2.0).
  - Optional corpus manifest for batch processing (schema v1.0).
- Outputs:
  - QCReport JSON (single-file analysis).
  - Batch summary JSON/Markdown and a one-page HTML report.
  - Reproducibility Markdown (optional).

### Analysis pipeline (normative)
The analysis pipeline MUST follow this order and use the specified algorithms:

1) Decode and normalize audio
- Decode via `soundfile` when available; on failure, fall back to `ffmpeg` if installed.
- Normalize to mono or stereo as per channel policy.
- Record decode warnings and backend selection.

2) Channel policy application
Supported policies:
- `mono`: downmix any input to mono by averaging channels.
- `stereo_average`: for stereo, use (L+R)/2; for mono, passthrough.
- `mid_only`: same computation as `stereo_average` but labeled differently for audit.
- `per_channel`: analyze left and right separately; only allowed in exploratory mode.

3) Optional resampling
If `analysis_lock.resample_fs_hz` is set, resample using linear interpolation.

4) Long-term PSD (LTPSD)
Compute Welch PSD with:
- Window: Hann
- One-sided correction
- Power-per-Hz normalization
- Per-frame PSD in dB, then mean and variance across frames

5) Frequency-grid alignment
Linearly interpolate input PSD and variance to the profile frequency grid with edge clamping.

6) Optional smoothing
If `analysis_lock.smoothing.type` is `octave_fraction`, apply octave-fraction averaging over the profile grid.

7) Deviation curve
Compute per-bin deviation: `delta_db = input_mean_db - reference_mean_db`.

8) Band metrics
For each band:
- Mean deviation: df-weighted average of `delta_db` within the band.
- Max deviation: maximum absolute deviation within the band.
- Variance ratio: df-weighted input variance divided by reference variance.

9) Global metrics
- Spectral tilt: linear regression of dB vs log2(f) from 50 Hz to 16 kHz.
- Tilt deviation: input tilt minus reference tilt.
- True peak: 4x oversampled reconstruction with Kaiser-windowed sinc FIR.
- Integrated loudness (LUFS): ITU-R BS.1770 via `pyloudnorm` when available; otherwise recorded as unavailable.

10) Threshold evaluation
Apply profile thresholds to band and global metrics. See Decision logic below.

11) QCReport construction
Assemble report metadata, metrics, decisions, confidence, and integrity hash.

### Decision logic (formal)
Given band metrics and global metrics:
- Band mean deviation status: compare absolute mean deviation to `band_mean` pass/warn limits.
- Band max deviation status: compare absolute max deviation to `band_max` pass/warn limits.
- Variance ratio status: compare ratio to `variance_ratio` pass/warn limits (higher is worse).
- Tilt deviation status: compare absolute tilt deviation to `tilt` pass/warn limits.
- True peak status: compare measured true peak to `true_peak` pass/warn limits when enabled.

Overall status:
- FAIL if any metric status is FAIL.
- WARN if the number of WARN metrics in bands meets or exceeds `warn_if_warn_band_count_at_least`.
- PASS otherwise.

### Confidence assessment (formal)
Confidence is evaluated independently of pass/warn/fail and can downgrade confidence to WARN if any of these conditions are true:
- Zero-length audio or non-positive effective duration.
- Effective duration shorter than 0.5 seconds.
- Silence ratio >= 0.5 based on RMS frames.
- Truncated decode warnings reported by the backend.
- Resampling applied (analysis used a resampled version).

### Hashing and quantization (formal)
- Canonical JSON uses UTF-8, lexicographically sorted keys, and `,`/`:` separators.
- The QCReport hash is computed over the canonical JSON of the report excluding the `integrity` object.
- Quantization prior to hashing:
  - dB curves and deviations: 0.01
  - variance and ratios: 0.001
  - tilt: 0.001 dB per octave
  - LUFS and true peak: 0.01

## File format specifications

### QCReport JSON (output)
QCReport is a JSON object with these top-level fields:
- `schema_version`: string, currently `1.0`.
- `report_id`: unique report identifier.
- `created_utc`: ISO-8601 UTC timestamp.
- `engine`: name, version, and build metadata.
- `input`: file path, hashes, sample rate, channels, duration, decode backend, decode warnings.
- `profile`: profile metadata, analysis lock hash, and algorithm IDs.
- `analysis`: analysis parameters, algorithm registry, smoothing, bands, normalization, and silence gate.
- `metrics`: frequency grid, LTPSD mean/variance, deviation curve, band metrics, global metrics.
- `decisions`: overall status and metric-level decisions.
- `confidence`: confidence status and reasons.
- `integrity`: hash and signature placeholder.


Field details:
- `input.file_hash_sha256` is the SHA-256 hash of the source file; if hashing fails it is set to 64 zeros.
- `input.decoded_pcm_hash_sha256` is the SHA-256 hash of the decoded PCM byte stream.
- `engine.build.deps` includes numpy and ffmpeg version strings; `hash_sha256` values are placeholders.
- `analysis.normalization.loudness.measured_lufs_i` is set to -100.0 when loudness cannot be measured.
- `analysis.silence_gate.enabled` is currently always false; the fields still report computed silence statistics.
- `confidence.status` is currently `pass` or `warn`; `fail` is reserved for future extensions.

The structure matches the implementation in `spectraqc/reporting/qcreport.py` and `spectraqc/cli/main.py`.

### Reference profile schema (v2.0)
A reference profile MUST conform to `docs/reference_profile_v2.schema.json`. Key sections:
- `profile`: name, kind, and version.
- `frequency_grid`: one-sided FFT grid in Hz.
- `reference_curves`: mean and variance curves, with optional percentiles.
- `bands`: list of frequency bands for aggregation.
- `analysis_lock`: FFT size, hop size, window, smoothing, channel policy, normalization policy.
- `threshold_model`: pass/warn rules and aggregation behavior.
- `integrity`: profile hash and signature placeholder.

Usefulness: profiles lock the analysis parameters and expected spectral shape so that comparisons are meaningful and repeatable.

### Corpus manifest schema (v1.0)
A corpus manifest MUST conform to `docs/corpus_manifest_v1.schema.json`. Key sections:
- `root_dir`: base directory for relative file paths.
- `exclusions`: glob patterns to exclude matching files.
- `files`: list of file entries with `path`, optional `hash_sha256`, optional `duration_s`, and optional `exclude`.

Usefulness: manifests provide a stable, auditable list of inputs for batch analysis and profile construction.

## Command reference

### Global CLI
- Command: `spectraqc`
- Purpose: entry point for analysis and inspection workflows.
- Usefulness: provides stable, scriptable operations with explicit exit codes.

Global option:
- `--version`
  - Spec: prints `spectraqc <version>` and exits successfully.
  - Usefulness: helps auditors capture the exact software version used for a run.

Exit codes:
- `0`: PASS
- `10`: WARN
- `20`: FAIL
- `2`: bad arguments or invalid usage
- `3`: input decode error
- `4`: profile load/validation error
- `5`: internal error

Stdout/stderr discipline:
- Stdout is for primary outputs (reports and summaries).
- Stderr is for errors and diagnostic messages.

### `spectraqc analyze`
- Purpose: produce a full QCReport for a single audio file.
- Usefulness: yields a complete, structured report suitable for archival and audit.

Usage:
```
spectraqc analyze <audio_path> --profile <ref.json> [options]
```

Arguments and options:
- `audio_path` (positional)
  - Spec: path to the audio file to analyze.
  - Usefulness: defines the single analysis target.
- `--profile`, `-p` (required)
  - Spec: path to the reference profile JSON.
  - Usefulness: determines the expected spectral profile and thresholds.
- `--mode`, `-m` (default: `compliance`, choices: `compliance`, `exploratory`)
  - Spec: selects the analysis mode. `per_channel` policy is only allowed in `exploratory`.
  - Usefulness: supports stricter compliance checks or exploratory multi-channel runs.
- `--out`, `-o`
  - Spec: write QCReport JSON to this path instead of stdout.
  - Usefulness: enables file-based pipelines and archival storage.
- `--repro-md`
  - Spec: write a reproducibility Markdown document to this path.
  - Usefulness: captures the exact inputs and algorithm IDs for future reruns.

### `spectraqc validate`
- Purpose: run analysis and print a minimal pass/warn/fail summary.
- Usefulness: supports fast gating in CI or batch scripts without full JSON output.

Usage:
```
spectraqc validate <audio_path> --profile <ref.json> [options]
```

Arguments and options:
- `audio_path` (positional)
  - Spec: path to the audio file to validate.
  - Usefulness: defines the single validation target.
- `--profile`, `-p` (required)
  - Spec: path to the reference profile JSON.
  - Usefulness: provides the thresholds and reference curves for validation.
- `--fail-on` (default: `fail`, choices: `fail`, `warn`)
  - Spec: returns non-zero on FAIL only, or on WARN and FAIL.
  - Usefulness: lets pipelines treat warnings as failures when needed.

### `spectraqc inspect-ref`
- Purpose: print a human-readable summary of a reference profile.
- Usefulness: quick sanity check for band definitions and thresholds.

Usage:
```
spectraqc inspect-ref --profile <ref.json>
```

Arguments and options:
- `--profile`, `-p` (required)
  - Spec: path to the reference profile JSON.
  - Usefulness: selects the profile to inspect.

### `spectraqc batch`
- Purpose: analyze a folder or corpus manifest in parallel.
- Usefulness: enables corpus-scale QC and aggregate reporting.

Usage:
```
spectraqc batch [options]
```

Arguments and options:
- `--folder`
  - Spec: folder containing audio files to analyze.
  - Usefulness: simple batch input without a manifest.
- `--manifest`
  - Spec: corpus manifest JSON path.
  - Usefulness: provides a stable, auditable list of files and exclusions.
- `--profile`, `-p` (required)
  - Spec: path to the reference profile JSON.
  - Usefulness: defines the expected spectral target and thresholds.
- `--mode`, `-m` (default: `compliance`, choices: `compliance`, `exploratory`)
  - Spec: analysis mode, same semantics as `analyze`.
  - Usefulness: supports strict or exploratory batch runs.
- `--out-dir`
  - Spec: directory for QCReport outputs and batch reports.
  - Usefulness: ensures batch outputs are co-located for review.
- `--summary-json` (default: `batch-summary.json`)
  - Spec: filename for summary JSON, written under `--out-dir`.
  - Usefulness: provides machine-readable aggregate metrics and counts.
- `--summary-md` (default: `batch-summary.md`)
  - Spec: filename for summary Markdown, written under `--out-dir`.
  - Usefulness: quick human-readable summary for logs or reviews.
- `--report-md` (default: `batch-report.md`)
  - Spec: filename for one-page Markdown report.
  - Usefulness: shareable batch report for stakeholders.
- `--report-html` (default: `batch-report.html`)
  - Spec: filename for one-page HTML report with embedded viewer.
  - Usefulness: interactive inspection of per-file QCReports.
- `--repro-md` (default: `batch-repro.md`)
  - Spec: filename for reproducibility Markdown.
  - Usefulness: records the exact batch run configuration.
- `--recursive`
  - Spec: if set, recursively scan subfolders when using `--folder`.
  - Usefulness: supports nested corpus layouts.
- `--workers` (default: `cpu_count - 1`)
  - Spec: number of parallel worker processes.
  - Usefulness: balances throughput and CPU usage for large batches.

Batch file discovery:
- Only files with extensions `.wav`, `.flac`, `.aiff`, `.aif`, `.mp3` are included when using `--folder`.

Output behavior:
- If any batch report outputs are requested, `--out-dir` is required.
- QCReport files are written as `<audio_stem>.qcreport.json` under `--out-dir`.

## Development scripts

### `scripts/synth_vectors.py`
- Purpose: generate deterministic synthetic audio vectors for validation.
- Usefulness: provides known signals to validate analysis and thresholds.
- Options: none (run as `python scripts/synth_vectors.py`).

### `scripts/build_dev_ref.py`
- Purpose: build a reference profile from a corpus manifest.
- Usefulness: produces a profile for development and validation workflows.

Usage:
```
python scripts/build_dev_ref.py --manifest <manifest.json> [options]
```

Options:
- `--manifest` (required)
  - Spec: path to corpus manifest JSON.
  - Usefulness: defines the corpus used to build the profile.
- `--out`
  - Spec: output path for the profile JSON; defaults to `validation/profiles/<name>.ref.json`.
  - Usefulness: lets you store generated profiles in custom locations.
- `--name` (default: `streaming_generic_v1`)
  - Spec: profile name.
  - Usefulness: identifies the profile in reports and audits.
- `--kind` (default: `streaming`)
  - Spec: profile kind (`broadcast`, `streaming`, `archive`, `custom`).
  - Usefulness: provides contextual classification for consumers.

## Known limitations
- Loudness measurement depends on the optional `pyloudnorm` library. If it is not installed, LUFS is reported as unavailable.
- MP3 decoding requires either `soundfile` support or `ffmpeg` on the system.
- Confidence modeling is conservative and based on decode and silence checks only.

## References
- CLI and QCReport spec: `docs/SpectraQC_V1_Full_Implementation.md`
- V2 scope: `docs/V2_SCOPE.md`
- Profile versioning rules: `docs/profile_versioning.md`
- Reference profile schema: `docs/reference_profile_v2.schema.json`
- Corpus manifest schema: `docs/corpus_manifest_v1.schema.json`


