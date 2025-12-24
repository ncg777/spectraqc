import numpy as np
import pyloudnorm as pyln

def integrated_lufs_mono(x, fs):
    meter = pyln.Meter(fs)
    return float(meter.integrated_loudness(np.asarray(x)))
