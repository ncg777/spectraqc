from __future__ import annotations
from spectraqc.types import AudioBuffer, LongTermPSD
from spectraqc.dsp.psd import welch_psd_db


def compute_ltpsd(audio: AudioBuffer, nfft: int, hop: int) -> LongTermPSD:
    """Compute long-term PSD from audio buffer using Welch's method."""
    freqs, mean_db, var_db2 = welch_psd_db(audio.samples, audio.fs, nfft=nfft, hop=hop)
    return LongTermPSD(freqs=freqs, mean_db=mean_db, var_db2=var_db2)
