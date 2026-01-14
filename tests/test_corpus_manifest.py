from __future__ import annotations

import json

from spectraqc.corpus.manifest import load_corpus_manifest, validate_manifest_entry


def test_load_corpus_manifest_and_validate(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"data")
    manifest = {
        "schema_version": "1.0",
        "root_dir": str(tmp_path),
        "exclusions": [],
        "files": [{"path": "audio.wav"}]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, entries, _ = load_corpus_manifest(str(manifest_path))
    assert entries[0].path == audio_path.resolve()
    validate_manifest_entry(entries[0])
