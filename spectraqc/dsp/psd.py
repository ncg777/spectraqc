import numpy as np
from spectraqc.dsp.windowing import hann, window_power_norm

def welch_psd_db(x, fs, nfft=4096, hop=2048):
    w = hann(nfft)
    U = window_power_norm(w)
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    has_nyq = nfft % 2 == 0
    gain = np.ones_like(freqs)
    gain[1:-1 if has_nyq else None] = 2.0
    frames = []
    for i in range(0, len(x)-nfft+1, hop):
        seg = x[i:i+nfft] * w
        X = np.fft.rfft(seg)
        P = (np.abs(X)**2) / (fs*nfft*U)
        P *= gain
        frames.append(10*np.log10(np.maximum(P,1e-20)))
    D = np.stack(frames)
    return freqs, D.mean(0), D.var(0)
