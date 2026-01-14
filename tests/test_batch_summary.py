from __future__ import annotations

import json
from pathlib import Path

from spectraqc.reporting.batch_summary import build_batch_summary
from spectraqc.utils.hashing import sha256_hex_canonical_json


def _load_fixture_results():
    fixture_path = Path(__file__).parent / "fixtures" / "batch_reports.json"
    reports = json.loads(fixture_path.read_text(encoding="utf-8"))
    results = []
    for report in reports:
        status = report.get("decisions", {}).get("overall_status", "error")
        results.append((report.get("input", {}).get("path", "unknown"), status, None, report))
    return results


def test_batch_summary_deterministic_checksum():
    results = _load_fixture_results()
    summary = build_batch_summary(results, generated_utc="2024-01-03T00:00:00Z")
    summary_repeat = build_batch_summary(results, generated_utc="2024-01-03T00:00:00Z")
    assert summary["checksum"]["corpus_checksum_sha256"] == summary_repeat["checksum"]["corpus_checksum_sha256"]


def test_batch_summary_kpis_and_hash_payload():
    results = _load_fixture_results()
    summary = build_batch_summary(results, generated_utc="2024-01-03T00:00:00Z")
    counts = summary["totals"]["status_counts"]
    assert counts["pass"] == 1
    assert counts["warn"] == 1
    assert summary["kpis"]["pass_rate"] == 0.5
    assert summary["kpis"]["mean_tilt_deviation_db_per_oct"] == 0.4
    band_rate = summary["band_failure_rates"][0]
    assert band_rate["band_name"] == "low"
    assert band_rate["warn_or_fail_rate"] == 0.5

    audio_hashes = sorted([
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    ])
    profile_hashes = ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
    analysis_lock_hashes = ["cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"]
    payload = {
        "audio_hashes": audio_hashes,
        "profile_hashes": profile_hashes,
        "analysis_lock_hashes": analysis_lock_hashes,
    }
    assert summary["checksum"]["corpus_checksum_sha256"] == sha256_hex_canonical_json(payload)
