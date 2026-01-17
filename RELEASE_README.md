# SpectraQC

**Deterministic Spectral Quality Control for Audio**

SpectraQC verifies that audio files conform to declared reference profiles with reproducible, audit-friendly outputs. It provides technical consistency checks that can be rerun with the same inputs and produce the same decisions.

## Key Features

- **Deterministic**: Same inputs produce the same metrics and decisions
- **Auditable**: Structured JSON outputs with cryptographic hashes
- **Explainable**: Decisions include per-band and global metric details
- **Corpus-scale**: Batch analysis with aggregate reporting
- **Self-contained**: No installation required—single executable

## Getting Started

### Quick Start (Single File)

```bash
spectraqc analyze audio.wav --profile profile.ref.json --out report.json
```

### Quick Start (Batch)

```bash
spectraqc batch --folder ./audio --profile profile.ref.json --out-dir ./reports
```

### Launch the GUI

```bash
spectraqc gui
```

## Commands

| Command | Purpose |
|---------|---------|
| `analyze` | Analyze a single audio file against a reference profile |
| `validate` | Quick pass/warn/fail check for a single file |
| `batch` | Analyze multiple files with summary reports |
| `build-profile` | Create a reference profile from your audio corpus |
| `inspect-ref` | Display reference profile details |
| `repair` | Apply DSP corrections using a repair plan |
| `gui` | Launch the interactive web interface |

Run `spectraqc --help` or `spectraqc <command> --help` for detailed options.

## Building Reference Profiles

Create a profile from your reference audio:

```bash
# From a folder of reference recordings
spectraqc build-profile --folder ./reference_audio --name my_profile --out my_profile.ref.json

# From a corpus manifest (for audit trails)
spectraqc build-profile --manifest corpus.json --name broadcast_v1 --kind broadcast
```

## Understanding Results

SpectraQC produces structured QCReport JSON files containing:

- **Input metadata**: File hashes, sample rate, duration, decode info
- **Metrics**: Spectral analysis, deviation curves, band measurements
- **Decisions**: PASS / WARN / FAIL for each metric and overall
- **Confidence**: Warnings for short or silent content
- **Integrity**: Report hash for verification

Exit codes indicate overall status:
- `0` = PASS
- `10` = WARN  
- `20` = FAIL

## External Dependencies

- **ffmpeg** (optional but recommended): Required for loudness measurement (ITU-R BS.1770) and MP3 decoding fallback. Install from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure it's in your PATH.

## Documentation

See the `docs/` folder for detailed documentation:

- [USER_GUIDE.md](docs/USER_GUIDE.md) — Practical walkthrough for teams
- [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) — Complete command reference
- [ONBOARDING_WORKSHOP.md](docs/ONBOARDING_WORKSHOP.md) — Training slides

## Support

For issues, feature requests, or questions, visit the GitHub repository.

## License

See LICENSE file for details.
