import numpy as np

def true_peak_dbtp_mono(x, fs, oversample=4):
    x = np.asarray(x, dtype=np.float64)
    if oversample == 1:
        return 20*np.log10(np.max(np.abs(x))+1e-30)
    y = np.zeros(len(x)*oversample)
    y[::oversample] = x
    h = np.sinc(np.linspace(-oversample, oversample, 63))
    h /= h.sum()
    z = np.convolve(y, h, mode="same")
    return 20*np.log10(np.max(np.abs(z))+1e-30)
