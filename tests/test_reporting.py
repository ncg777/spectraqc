from __future__ import annotations

import numpy as np

from spectraqc.reporting.qcreport import build_qcreport_dict
from spectraqc.utils.hashing import sha256_hex_canonical_json


def test_build_qcreport_hash_and_quantization():
    freqs = np.array([0.0, 1.0, 2.0])
    mean_db = np.array([0.004, 0.015, 0.021])
    var_db2 = np.array([0.0004, 0.0014, 0.0016])
    delta = np.array([0.004, -0.015, 0.021])
    report = build_qcreport_dict(
        engine={"name": "spectraqc", "version": "0"},
        input_meta={"path": "file.wav"},
        profile={"name": "p"},
        analysis={"report_id": "r", "created_utc": "now"},
        freqs_hz=freqs,
        ltpsd_mean_db=mean_db,
        ltpsd_var_db2=var_db2,
        delta_mean_db=delta,
        band_metrics=[
            {"band_name": "band", "f_low_hz": 0, "f_high_hz": 1, "mean_deviation_db": 0.004,
             "max_deviation_db": 0.015, "variance_ratio": 1.2345}
        ],
        global_metrics={"spectral_tilt_db_per_oct": 0.0004, "tilt_deviation_db_per_oct": 0.0006},
        decisions={"overall_status": "pass"},
        confidence={"status": "pass"}
    )
    assert report["metrics"]["ltpsd"]["mean_db"][0] == 0.0
    tmp = dict(report)
    tmp.pop("integrity", None)
    assert report["integrity"]["qcreport_hash_sha256"] == sha256_hex_canonical_json(tmp)
