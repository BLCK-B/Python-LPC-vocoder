import numpy as np
from scipy.fftpack import fft, ifft


def FFTautocorr(x):
    normalised = (x - np.average(x)) / np.std(x)
    fftRes = fft(normalised)
    powerSpDen = np.absolute(fftRes) ** 2
    pi = ifft(powerSpDen)
    result_length = int(len(normalised) / 2)
    return np.real(pi)[:result_length] / len(normalised)
