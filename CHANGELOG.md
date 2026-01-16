# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-01

### Added
- Deterministic spectral QC engine with reproducible hashing, QCReport JSON outputs, and batch summary/HTML reporting for audit-friendly workflows.
- V1 CLI command suite: `analyze`, `validate`, `inspect-ref`, `build-profile`, `batch`, and `gui` (plus global `--version`) with documented exit codes and stdout/stderr discipline.
- Repair workflow support via `spectraqc repair` with composable repair plans and recommended presets for archival and streaming outputs.
- Reference profile (v2) and corpus manifest (v1) schemas to standardize analysis inputs and batch manifests.

### Known Limitations
- Loudness measurement depends on `ffmpeg` (via `ebur128`); when unavailable, LUFS metrics are reported as unavailable.
- MP3 decoding requires `soundfile` support or `ffmpeg` on the system.
- Confidence modeling is conservative and currently based on decode/silence checks only.
- Input file hashing fields are placeholders (all zeros) until I/O hashing is finalized.
- WAV decoding is limited to PCM int16/int32 in V1; float WAV and other formats remain to be extended.
- True peak measurement uses OS4 + sinc FIR; a libebur128 implementation may be needed for strict parity.
- Confidence model is a placeholder pending richer checks (silence gate, decode errors, clipped audio detection).
