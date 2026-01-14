from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from spectraqc.cli.main import cmd_repair
from tests.conftest import build_profile_dict, write_profile


def test_cmd_repair_writes_outputs(tmp_path):
    fs = 48000
    t = np.arange(0, 0.5, 1.0 / fs)
    x = 1.1 * np.sin(2 * np.pi * 60.0 * t)
    x = np.clip(x, -1.0, 1.0)
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, x, fs)

    profile_path = write_profile(tmp_path, build_profile_dict())

    plan = {
        "steps": [
            {"name": "dehum", "params": {"hum_freq_hz": 60.0, "harmonics": 1}},
            {"name": "declip", "params": {"clip_threshold": 0.98}},
        ]
    }
    plan_path = tmp_path / "repair_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    out_audio = tmp_path / "output.wav"
    out_report = tmp_path / "report.json"
    args = SimpleNamespace(
        audio_path=str(audio_path),
        profile=str(profile_path),
        repair_plan=str(plan_path),
        out=str(out_audio),
        report=str(out_report),
    )

    exit_code = cmd_repair(args)
    assert exit_code in {0, 10, 20}
    assert out_audio.exists()
    assert out_report.exists()

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert "repair" in report
    assert report["repair"]["steps"]
    assert report["repair"]["metrics"]["before"]["deviation_curve_db"]

    repaired, _ = sf.read(out_audio, always_2d=True, dtype="float64")
    assert repaired.shape[0] == x.shape[0]
