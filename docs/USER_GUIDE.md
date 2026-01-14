# SpectraQC User Guide

This guide is a practical, structured walkthrough for teams who need deterministic, auditable spectral quality control at scale. It is written to support institutional workflows and repeatable results.

## 1) What SpectraQC Does

SpectraQC compares an audio file (or a corpus) to a declared reference profile. It measures spectral shape and stability, applies thresholds, and produces a QCReport with explicit PASS/WARN/FAIL decisions.

Key properties:
- **Deterministic**: the same inputs yield the same metrics and decisions (within floating‑point tolerance).
- **Auditable**: outputs are structured JSON with stable hashing.
- **Explainable**: decisions include per‑band and global metrics.
- **Corpus‑scale**: batch analysis and aggregate reporting.

## 1.1) Why Spectral Consistency Matters

SpectraQC is not asking for creative uniformity. It is designed for **technical consistency** within a defined workflow. In institutional environments, consistent spectra ensures:

- **Signal integrity across a pipeline**: capture and transfer issues (wrong EQ curve, damaged codec chain, faulty preamp, or missing low‑end due to a filter) are detectable when compared to a reference profile.
- **Policy compliance**: broadcast, archive, and streaming programs often mandate target spectral shapes and band limits to ensure intelligibility and minimize playback issues.
- **Fleet‑wide monitoring**: large collections reveal systemic drift (e.g., a microphone batch, an ingestion chain, or a digitization setup) only when you can compare to a stable reference.

Lower‑bound checks are not artistic constraints: they detect missing or compromised content (e.g., cut bass from a bad transfer, or missing high‑end from a faulty capture chain).

## 2) Installation

Use Python 3.11+ and install dependencies.

```bash
pip install -e .
```

Optional dependencies:
- `ffmpeg` (required for LUFS measurement and MP3 decoding fallback).

Check the CLI:

```bash
spectraqc --version
```

## 3) Inputs You Need

1) **Audio file(s)** in WAV, FLAC, AIFF, AIF, or MP3.
2) A **reference profile** (`.ref.json`) that defines the expected spectral target, bands, and thresholds.
3) *(Optional)* A **corpus manifest** (`.json`) for batch processing with an auditable file list.

## 4) Quick Start (Single File)

```bash
spectraqc analyze /path/to/audio.wav --profile /path/to/profile.ref.json --out report.qcreport.json
```

To generate a reproducibility doc:

```bash
spectraqc analyze /path/to/audio.wav --profile /path/to/profile.ref.json --repro-md report-repro.md
```

## 5) Quick Start (Batch)

Batch analyze a folder:

```bash
spectraqc batch --folder /path/to/audio --profile /path/to/profile.ref.json --out-dir /path/to/output
```

Batch analyze via manifest:

```bash
spectraqc batch --manifest /path/to/manifest.json --profile /path/to/profile.ref.json --out-dir /path/to/output
```

Batch outputs:
- `*.qcreport.json` per file
- `batch-summary.json`
- `batch-summary.md`
- `batch-report.md`
- `batch-report.html`
- `batch-repro.md`

## 6) Understanding the QCReport

The QCReport has these top-level sections:
- `input`: file paths, hashes, and decode metadata
- `analysis`: parameters and algorithm registry
- `metrics`: LTPSD, deviation curve, band/global metrics
- `decisions`: PASS/WARN/FAIL outcomes
- `confidence`: sanity checks and warnings
- `integrity`: content hash

For detailed structure, see `docs/SpectraQC_V1_Full_Implementation.md`.

## 7) Profiles and Thresholds

Profiles define:
- Frequency grid (`frequency_grid.freqs_hz`)
- Reference curves (`reference_curves.mean_db`, `var_db2`)
- Bands (`bands`)
- Analysis parameters (`analysis_lock`)
- Threshold model (`threshold_model`)

Use profiles to ensure repeatable comparisons. A profile is required for every run.

## 8) Analysis Modes

Mode controls channel policy and compliance behavior:
- `compliance`: single-channel decision with restricted policies
- `exploratory`: allows `per_channel` analysis for diagnostics

## 9) Troubleshooting

Common issues:
- **LUFS not measured**: install `ffmpeg` and re-run.
- **MP3 decode errors**: ensure `ffmpeg` is installed.
- **Profile validation errors**: check `docs/reference_profile_v2.schema.json`.

If output is unexpected, confirm:
- The profile grid matches your target frequency range.
- Bands are within the profile grid.
- `analysis_lock` matches your intended parameters.

## 10) Operational Recommendations

For institutional deployments:
- Version and archive profiles in source control.
- Store QCReports alongside source audio for audit trails.
- Record command lines or use `--repro-md` for formal reproducibility.
- Validate corpora using manifests to ensure input integrity.

## 11) Where to Go Next

- CLI reference: `docs/CLI_REFERENCE.md`
- Onboarding workshop deck: `docs/ONBOARDING_WORKSHOP.md`
- Profile schema: `docs/reference_profile_v2.schema.json`

## 12) Manual Browser Snapshot Test Plan (Report Viewer)

The CLI renders HTML with an embedded report viewer. Use the steps below to manually verify chart interactions and capture snapshots when reviewing UI changes.

1. Generate a batch HTML report (or open an existing one):
   ```bash
   spectraqc batch --folder /path/to/audio --profile /path/to/profile.ref.json --out-dir /tmp/spectraqc
   ```
   Open `/tmp/spectraqc/batch-report.html` in a browser.
2. In the report viewer, select a QCReport with populated charts.
3. For each chart (LTPSD, Deviation Curve, band/global bars), verify:
   - Zoom in/out buttons update the canvas.
   - Reset returns to the full-range view.
   - Scroll wheel zooms toward the cursor position.
   - Drag a selection box to zoom into a region.
   - Drag-to-pan (right click or shift-drag) moves the viewport.
4. Click **Full screen**:
   - Confirm the overlay shows a live canvas (not a static image).
   - Verify zoom/pan controls still work in the overlay.
   - Confirm the zoom level matches the inline chart when you toggle full screen.
5. Capture screenshots for any UI changes and attach them to review notes.
