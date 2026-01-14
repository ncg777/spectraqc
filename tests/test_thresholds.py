from __future__ import annotations

from spectraqc.thresholds.evaluator import evaluate
from spectraqc.types import FrequencyBand, BandMetrics, GlobalMetrics, Status


def test_evaluate_warn_count_only_bands():
    band = FrequencyBand("band", 20.0, 200.0)
    metrics = [
        BandMetrics(band=band, mean_deviation_db=1.5, max_deviation_db=1.0, variance_ratio=1.0)
    ]
    thresholds = {
        "band_mean_db": {"default": (1.0, 2.0)},
        "band_max_db": {"all": (2.0, 4.0)},
        "tilt_db_per_oct": (0.5, 1.0),
        "variance_ratio": (1.2, 1.5),
        "warn_if_warn_band_count_at_least": 2,
        "true_peak_dbtp": (-1.0, 0.0),
    }
    globals_metrics = GlobalMetrics(
        spectral_tilt_db_per_oct=0.0,
        tilt_deviation_db_per_oct=0.0,
        true_peak_dbtp=-0.5
    )
    decision = evaluate(metrics, globals_metrics, thresholds)
    assert decision.overall_status == Status.PASS


def test_evaluate_fail_on_global():
    band = FrequencyBand("band", 20.0, 200.0)
    metrics = [
        BandMetrics(band=band, mean_deviation_db=0.1, max_deviation_db=0.1, variance_ratio=1.0)
    ]
    thresholds = {
        "band_mean_db": {"default": (1.0, 2.0)},
        "band_max_db": {"all": (2.0, 4.0)},
        "tilt_db_per_oct": (0.5, 1.0),
        "variance_ratio": (1.2, 1.5),
        "warn_if_warn_band_count_at_least": 2,
        "true_peak_dbtp": (-1.0, 0.0),
    }
    globals_metrics = GlobalMetrics(
        spectral_tilt_db_per_oct=0.0,
        tilt_deviation_db_per_oct=0.0,
        true_peak_dbtp=0.5
    )
    decision = evaluate(metrics, globals_metrics, thresholds)
    assert decision.overall_status == Status.FAIL
