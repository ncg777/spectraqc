# Profile Versioning Rules

Profiles use semantic versioning: `MAJOR.MINOR.PATCH`.

## Versioning scope

The profile version reflects the profile data and analysis assumptions encoded in the `.ref.json` file. It is independent of the `spectraqc` package version.

### Hashing

The `integrity.profile_hash_sha256` is computed from the canonical JSON of the profile content (excluding the `integrity` object). Any change to hashed fields requires a hash update.

## Version bump rules

### PATCH (compatible, no behavioral change)

Increment PATCH when changes do not alter analysis outputs or decisions for the same input and algorithms.

- Metadata-only edits: `profile.name`, `profile.kind`, `profile.version` formatting fix.
- Documentation fields: `notes`, `corpus_stats` (when derived from unchanged inputs).
- Re-serialization changes (key ordering, whitespace) without content changes.
- Hash/signature updates only, if content is unchanged.

Hash changes: only if hashed content changes (usually no).

### MINOR (compatible, measurable behavior change)

Increment MINOR when changes alter numerical outputs but keep the same schema and intent.

- Threshold adjustments (pass/warn values, aggregation rules).
- Band definition tweaks (adjusted cutoffs, added/removed bands) when the analysis goal remains the same.
- Reference curves updated from an expanded/updated corpus using the same algorithm IDs and parameters.
- Smoothing parameters adjusted within the same algorithm (e.g., octave fraction change).
- Normalization policy changes that do not change enabled/disabled state (e.g., target LUFS change).

Hash changes: yes.

### MAJOR (breaking or incompatible semantics)

Increment MAJOR when a profile is not directly comparable to prior versions or requires code changes.

- Schema version change (e.g., `schema_version` bump).
- Algorithm registry changes that alter algorithm IDs or locked parameters.
- Changing analysis lock fundamentals (window type, PSD estimator, FFT size policy) that shift expected outputs.
- Normalization enablement changes (e.g., loudness normalization enabled/disabled, true-peak enablement changes).
- Channel policy changes that alter how multichannel audio is collapsed (mono vs stereo/mid/per-channel).
- Any change requiring a new interpretation of the profile by consumers.

Hash changes: yes.

## Examples

- Update only `profile.name`: PATCH (hash unchanged if only metadata fields change outside hashed content).
- Update band thresholds: MINOR + hash update.
- Switch loudness algorithm ID: MAJOR + hash update.
- Add new required field via schema: MAJOR + hash update.

## Profile comparison rule

Profiles are comparable if:
- Same `schema_version`.
- Same algorithm registry entries and parameters.
- Same channel policy.

If any of the above differ, treat comparison as incompatible.
