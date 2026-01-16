# SpectraQC CLI Reference

This reference describes CLI commands, arguments, expected outputs, and exit codes. It is intended to be concise and operational.

## Global Usage

```bash
spectraqc --version
spectraqc <command> [options]
```

Exit codes:
- `0`: PASS
- `10`: WARN
- `20`: FAIL
- `2`: bad arguments
- `3`: input decode error
- `4`: profile load/validation error
- `5`: internal error

## Command: `analyze`

Purpose: analyze a single file and produce a full QCReport.

```bash
spectraqc analyze <audio_path> --profile <ref.json> [options]
```

Options:
- `--profile`, `-p` **(required)**: reference profile JSON
- `--mode`, `-m`: `compliance` (default) or `exploratory`
- `--out`, `-o`: output path for QCReport JSON
- `--repro-md`: output path for reproducibility Markdown

Outputs:
- QCReport JSON to stdout or `--out`
- Exit code based on overall status

## Command: `validate`

Purpose: produce a short pass/warn/fail summary for a single file.

```bash
spectraqc validate <audio_path> --profile <ref.json> [options]
```

Options:
- `--profile`, `-p` **(required)**: reference profile JSON
- `--fail-on`: `fail` (default) or `warn`

Outputs:
- Summary text to stdout
- Exit code based on selected `--fail-on` policy

## Command: `inspect-ref`

Purpose: print a short, human-readable summary of a reference profile.

```bash
spectraqc inspect-ref --profile <ref.json>
```

Options:
- `--profile`, `-p` **(required)**: reference profile JSON

Outputs:
- Profile metadata, bands, and thresholds

## Command: `build-profile`

Purpose: build a reference profile from a manifest or folder.

```bash
spectraqc build-profile --manifest <manifest.json> [options]
spectraqc build-profile --folder <audio_dir> [options]
```

Options:
- `--manifest`: corpus manifest JSON path
- `--folder`: folder containing reference audio files
- `--recursive`: scan subfolders when using `--folder`
- `--out`: output path for the profile JSON
- `--name`: profile name (default: `streaming_generic_v1`)
- `--kind`: profile kind (`broadcast`, `streaming`, `archive`, `custom`)

Outputs:
- Reference profile JSON written to disk
- Summary of profile metadata to stdout

## Command: `batch`

Purpose: analyze a folder or manifest in parallel and produce batch summaries.

```bash
spectraqc batch [options]
```

Input options:
- `--folder`: folder containing audio files
- `--manifest`: corpus manifest JSON
- `--profile`, `-p` **(required)**: reference profile JSON

Output options:
- `--out-dir`: output directory for QCReports and batch reports
- `--summary-json`: summary JSON filename (default: `batch-summary.json`)
- `--summary-md`: summary Markdown filename (default: `batch-summary.md`)
- `--report-md`: one‑page Markdown report filename (default: `batch-report.md`)
- `--report-html`: one‑page HTML report filename (default: `batch-report.html`)
- `--repro-md`: reproducibility Markdown filename (default: `batch-repro.md`)

Other options:
- `--mode`, `-m`: `compliance` (default) or `exploratory`
- `--recursive`: scan subfolders when using `--folder`
- `--workers`: parallel workers (default: `cpu_count - 1`)

Behavior notes:
- If any batch reports are requested, `--out-dir` is required.
- QCReports are written as `<audio_stem>.qcreport.json` in `--out-dir`.

## Examples

Single‑file QC:
```bash
spectraqc analyze audio.wav --profile profile.ref.json --out audio.qcreport.json
```

Batch QC with folder scan:
```bash
spectraqc batch --folder ./audio --profile profile.ref.json --out-dir ./reports --recursive
```

Batch QC with manifest:
```bash
spectraqc batch --manifest manifest.json --profile profile.ref.json --out-dir ./reports
```
