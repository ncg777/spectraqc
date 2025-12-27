"""Corpus manifest ingestion and validation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import fnmatch
from typing import Any
from spectraqc.utils.hashing import sha256_hex_file


@dataclass(frozen=True)
class CorpusEntry:
    """Validated corpus entry."""
    path: Path
    hash_sha256: str | None
    duration_s: float | None
    exclude: bool = False


def _as_path(base: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_corpus_manifest(path: str) -> tuple[Path, list[CorpusEntry], list[str]]:
    """Load and validate a corpus manifest JSON."""
    import json
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    errors: list[str] = []
    warnings: list[str] = []

    schema_version = j.get("schema_version", "1.0")
    if schema_version != "1.0":
        errors.append("manifest.schema_version must be '1.0'.")

    root_dir = j.get("root_dir")
    base = Path(root_dir).resolve() if root_dir else manifest_path.parent

    exclusions = j.get("exclusions", [])
    if exclusions and not isinstance(exclusions, list):
        errors.append("manifest.exclusions must be a list of glob patterns.")
        exclusions = []

    files = j.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest.files must be a non-empty list.")
        files = []

    entries: list[CorpusEntry] = []
    for i, fobj in enumerate(files):
        if not isinstance(fobj, dict):
            errors.append(f"files[{i}] must be an object.")
            continue
        path_value = fobj.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"files[{i}].path must be a non-empty string.")
            continue
        abs_path = _as_path(base, path_value)
        rel_path = str(Path(path_value))
        excluded = bool(fobj.get("exclude", False))
        if any(fnmatch.fnmatch(rel_path, pat) for pat in exclusions):
            excluded = True
        hash_sha256 = fobj.get("hash_sha256")
        if hash_sha256 is not None and not isinstance(hash_sha256, str):
            errors.append(f"files[{i}].hash_sha256 must be a string if provided.")
        duration_s = fobj.get("duration_s")
        if duration_s is not None and not isinstance(duration_s, (int, float)):
            errors.append(f"files[{i}].duration_s must be a number if provided.")
        entries.append(CorpusEntry(
            path=abs_path,
            hash_sha256=hash_sha256,
            duration_s=float(duration_s) if duration_s is not None else None,
            exclude=excluded
        ))

    if errors:
        raise ValueError("; ".join(errors))

    return base, entries, warnings


def validate_manifest_entry(
    entry: CorpusEntry,
    *,
    duration_s: float | None = None,
    duration_tolerance_s: float = 0.05
) -> None:
    """Validate file hash and optional duration against manifest."""
    if not entry.path.exists():
        raise ValueError(f"Manifest file not found: {entry.path}")
    if entry.hash_sha256:
        actual = sha256_hex_file(str(entry.path))
        if actual != entry.hash_sha256:
            raise ValueError(f"Hash mismatch for {entry.path.name}")
    if entry.duration_s is not None and duration_s is not None:
        if abs(duration_s - entry.duration_s) > duration_tolerance_s:
            raise ValueError(f"Duration mismatch for {entry.path.name}")
