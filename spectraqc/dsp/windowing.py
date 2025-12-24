import numpy as np
def hann(n): return np.hanning(n).astype(np.float64)
def window_power_norm(w): return float(np.mean(w**2))
