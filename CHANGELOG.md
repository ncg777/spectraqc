# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-17

### Added
- Deterministic spectral QC engine with reproducible hashing, QCReport JSON outputs, and batch summary/HTML reporting for audit-friendly workflows.
- V1 CLI command suite: `analyze`, `validate`, `inspect-ref`, `build-profile`, `batch`, `repair`, and `gui` with documented exit codes and stdout/stderr discipline.
- Standalone executable distribution via PyInstaller—no Python installation required for end users.
- Repair workflow support via `spectraqc repair` with composable repair plans and recommended presets for archival and streaming outputs.
- Interactive web GUI via `spectraqc gui` for single-file analysis and profile building.
- Reference profile (v1) and corpus manifest (v1) schemas to standardize analysis inputs and batch manifests.
- Build scripts for executable packaging (`build_exe.py`) and GitHub release preparation (`build_release.py`).
- Comprehensive documentation: USER_GUIDE.md, CLI_REFERENCE.md, ONBOARDING_WORKSHOP.md.

### Known Limitations
- Loudness measurement depends on `ffmpeg` (via `ebur128`); when unavailable, LUFS metrics are reported as unavailable.
- MP3 decoding requires `soundfile` support or `ffmpeg` on the system.
- Confidence modeling is conservative and currently based on decode/silence checks only.
