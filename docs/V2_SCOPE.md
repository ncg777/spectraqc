# SpectraQC V2 — Scope Definition

## Purpose

This document defines the **explicit scope and non-scope** of SpectraQC V2.

Its purpose is to:
- prevent scope creep,
- eliminate ambiguity,
- ensure engineering effort aligns with pilot and institutional needs.

This document is **normative**.  
Any deviation requires explicit revision and commit.

---

## Product Intent (V2)

SpectraQC V2 is a **deterministic, audit-ready audio corpus quality and consistency verification tool**.

It is designed for:
- institutional archives,
- public broadcasters,
- libraries and universities,
- large-scale digitization and preservation projects,
- AI / research audio corpora requiring reproducibility.

SpectraQC V2 answers the question:

> *“Can we objectively verify that this audio corpus is technically valid, internally consistent, and conforms to a declared reference profile — in a way that can be reproduced in the future?”*

---

## In-Scope Capabilities (V2)

### 1. Audio Ingestion & Decoding
- Robust decoding of common archival formats:
  - WAV
  - FLAC
  - AIFF
  - MP3 (via FFmpeg or equivalent)
- Multichannel handling with explicit, declared policies:
  - mono
  - stereo average
  - mid-only
  - per-channel (analysis only)
- Capture and record decode warnings and anomalies.

---

### 2. Deterministic Signal Analysis
- Long-term spectral analysis using locked algorithms.
- Explicit, declared:
  - FFT size
  - hop size
  - window function
  - PSD estimator
- Deterministic numerical behavior across runs and machines (within floating-point tolerance).

---

### 3. Standards-Based Metrics
Where standards exist, V2 uses **reference-grade implementations**:
- Integrated loudness (ITU-R BS.1770)
- True peak level (oversampled, reference implementation)

All metrics:
- declare algorithm IDs,
- lock parameters,
- are recorded in outputs.

---

### 4. Reference Profile System
- Reference profiles (`.ref.json`) are first-class artifacts.
- Profiles are built from real audio corpora using a declared corpus manifest.
- Profiles include:
  - frequency grid
  - reference curves
  - variability envelopes
  - band definitions
  - thresholds
  - normalization policies
- Profiles are versioned and hash-stable.

---

### 5. Verification & Decision Logic
- Verification against a reference profile produces:
  - PASS / WARN / FAIL decisions
  - explicit metric-level explanations
- Confidence is modeled **separately** from pass/fail.
- Uncertainty sources (e.g. silence, short duration, decode issues) reduce confidence, not correctness.

---

### 6. Batch & Corpus-Scale Operation
- Batch processing of large collections (folders or manifests).
- Safe parallel execution.
- Resume-on-failure behavior.
- Corpus-level aggregation:
  - counts
  - distributions
  - dominant failure causes.

---

### 7. Outputs & Deliverables
- Machine-readable, canonical JSON outputs (QCReport).
- Stable hashing for reproducibility and audit use.
- Human-readable summary reports (Markdown or HTML).
- Reproducibility notes sufficient for future reruns.

---

## Explicitly Out of Scope (V2)

The following are **explicitly excluded** from V2:

### ❌ Creative / Subjective Features
- Audio enhancement or correction
- Mastering, EQ, or loudness normalization
- Subjective “quality scores”
- Genre-aware or stylistic analysis

### ❌ User Interface Features
- Graphical user interface (GUI)
- Real-time visualization
- Interactive dashboards

### ❌ AI / Machine Learning
- ML-based analysis
- Learned reference models
- Adaptive or self-updating thresholds

### ❌ Deployment Models
- SaaS / hosted services
- Cloud-only execution
- Continuous background monitoring

### ❌ Format / Media Scope
- Video analysis
- Embedded metadata editing
- Non-audio signals

---

## Design Principles (V2)

SpectraQC V2 adheres to the following principles:

1. **Determinism over convenience**
2. **Auditability over aesthetics**
3. **Verification over optimization**
4. **Schemas over heuristics**
5. **Explainability over opacity**

Any proposed feature that violates these principles is out of scope.

---

## Success Criteria (V2)

SpectraQC V2 is considered complete when it can:

- Process a real institutional audio corpus end-to-end
- Build a reproducible reference profile from that corpus
- Produce explainable, deterministic verification results
- Be rerun at a later time with identical outputs
- Generate artifacts suitable for audits and long-term preservation workflows

---

## Change Control

Any change to this scope requires:
1. Editing this document
2. Recording the rationale
3. Explicit commit
4. Agreement that the change does not invalidate V2 goals

---

## Status

- [ ] Reviewed
- [ ] Accepted
- [ ] Committed
- [ ] Phase 0 closed
