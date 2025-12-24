"""
SpectraQC - Spectral Quality Control Tool

A tool for analyzing audio spectral characteristics against reference profiles.
"""
from spectraqc.version import __version__
from spectraqc.types import (
    Status,
    AudioBuffer,
    FrequencyBand,
    LongTermPSD,
    BandMetrics,
    GlobalMetrics,
    ThresholdResult,
    BandDecision,
    ProgramDecision,
    ReferenceProfile,
)

__all__ = [
    "__version__",
    "Status",
    "AudioBuffer",
    "FrequencyBand",
    "LongTermPSD",
    "BandMetrics",
    "GlobalMetrics",
    "ThresholdResult",
    "BandDecision",
    "ProgramDecision",
    "ReferenceProfile",
]