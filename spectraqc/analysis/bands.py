from __future__ import annotations
import numpy as np
from spectraqc.types import FrequencyBand


def band_mask(freqs: np.ndarray, band: FrequencyBand) -> np.ndarray:
    """Return boolean mask for frequencies within a band."""
    return (freqs >= band.f_low) & (freqs < band.f_high)


def default_streaming_bands() -> list[FrequencyBand]:
    """Return default frequency bands for streaming audio QC."""
    return [
        FrequencyBand("sub", 20, 60),
        FrequencyBand("bass", 60, 200),
        FrequencyBand("low_mid", 200, 800),
        FrequencyBand("mid", 800, 3000),
        FrequencyBand("high_mid", 3000, 8000),
        FrequencyBand("high", 8000, 16000),
        FrequencyBand("air", 16000, 20000),
    ]
