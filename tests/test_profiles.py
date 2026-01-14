from __future__ import annotations

import pytest

from spectraqc.profiles.loader import load_reference_profile
from spectraqc.profiles.validator import validate_reference_profile_dict

from tests.conftest import build_profile_dict, write_profile


def test_validate_reference_profile_accepts_log_hz(tmp_path):
    profile = build_profile_dict(smoothing={"type": "log_hz", "log_hz_bins_per_octave": 12})
    validate_reference_profile_dict(profile)


def test_validate_reference_profile_rejects_missing_key():
    profile = build_profile_dict()
    profile.pop("bands")
    with pytest.raises(ValueError, match="missing key: bands"):
        validate_reference_profile_dict(profile)


def test_load_reference_profile_true_peak_threshold(tmp_path):
    profile = build_profile_dict(true_peak_enabled=True)
    path = write_profile(tmp_path, profile)
    loaded = load_reference_profile(str(path))
    assert "true_peak_dbtp" in loaded.thresholds
