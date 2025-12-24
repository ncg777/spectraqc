import wave
import numpy as np
from spectraqc.types import AudioBuffer

def load_wav_mono(path: str) -> AudioBuffer:
    with wave.open(path, "rb") as wf:
        fs = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV format")
    x = x.reshape(-1, n_ch).mean(axis=1)
    return AudioBuffer(x.astype(np.float64), fs, len(x)/fs)
