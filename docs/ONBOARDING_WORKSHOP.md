# SpectraQC Onboarding Workshop (Markdown Slides)

> Audience: QC engineers, archivists, broadcast operations, and institutional IT.

---

## Slide 1 — Welcome

**SpectraQC: Deterministic Spectral Quality Control**
- Purpose: repeatable, auditable QC for audio assets
- Outcome: confident PASS/WARN/FAIL decisions backed by metrics

---

## Slide 2 — Why Determinism Matters

- Re-runs must yield the same decisions
- Reports must withstand audits
- QC must be explainable to non‑specialists

---

## Slide 3 — Core Concepts

- **Reference Profile**: expected spectral shape + thresholds
- **QCReport**: full, structured analysis output
- **Decision**: PASS/WARN/FAIL per band and globally

---

## Slide 4 — System Inputs

1. Audio file(s)
2. Reference profile (`.ref.json`)
3. Optional corpus manifest (`.json`)

---

## Slide 5 — Single‑File Flow

```bash
spectraqc analyze audio.wav --profile profile.ref.json --out audio.qcreport.json
```

Outputs:
- QCReport JSON
- Optional reproducibility doc

---

## Slide 6 — Batch Flow

```bash
spectraqc batch --folder ./audio --profile profile.ref.json --out-dir ./reports
```

Outputs:
- QCReports per file
- Batch summary JSON + Markdown
- HTML batch report for stakeholders

---

## Slide 7 — What the QCReport Contains

- `input`: hashes, decode metadata
- `analysis`: locked parameters + algorithm registry
- `metrics`: LTPSD, reference curves (when supplied), deviation, band/global metrics
- `decisions`: PASS/WARN/FAIL results
- `confidence`: warnings for short/silent content
- `integrity`: report hash

---

## Slide 8 — Profiles in Practice

Profiles define:
- Frequency grid
- Reference curves
- Bands
- Thresholds
- Analysis lock (FFT, smoothing, channel policy)

Operational tip: version profiles and store them in source control.

---

## Slide 9 — Channel Policies

- `mono`: downmix to mono
- `stereo_average`: (L+R)/2
- `mid_only`: (L+R)/2 (audited label)
- `per_channel`: exploratory diagnostics

---

## Slide 10 — Smoothing Options

- `none`: raw grid
- `octave_fraction`: fractional‑octave averaging
- `log_hz`: log‑spaced bin smoothing

Choose based on audit policy and reference curve design.

---

## Slide 11 — Interpreting Decisions

- FAIL if any metric fails
- WARN if band warnings meet the profile threshold
- PASS otherwise

Decisions are reported with explicit metric values and thresholds.

---

## Slide 12 — Confidence Flags

Confidence can be downgraded if:
- Audio is too short
- Silence ratio is high
- Decode warnings exist
- Resampling was applied

---

## Slide 13 — Operational Checklist

- [ ] Use a consistent profile for each program
- [ ] Archive QCReports with source assets
- [ ] Use manifests for large corpora
- [ ] Review batch HTML reports for trends

---

## Slide 14 — Troubleshooting

Common fixes:
- Install `ffmpeg` for LUFS + MP3 support
- Validate profile against the schema
- Confirm bands are inside the frequency grid

---

## Slide 15 — Next Steps

- Review `docs/USER_GUIDE.md`
- Review `docs/CLI_REFERENCE.md`
- Pilot with a sample corpus and calibrate thresholds
