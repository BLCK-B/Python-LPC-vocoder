import numpy as np
from scipy.fftpack import fft, ifft


def FFTautocorr(x):
    xp = (x - np.average(x)) / np.std(x)
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    result_length = int(len(xp) / 2)
    return np.real(pi)[:result_length] / len(xp)
