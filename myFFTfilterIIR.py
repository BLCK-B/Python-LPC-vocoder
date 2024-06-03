import numpy as np
from scipy.fft import fft, ifft


def myFFTfilterIIR(coeff, x):
    # zero pad coeff to match x
    coeff = np.pad(coeff, (0, len(x) - len(coeff)), mode='constant')
    X = fft(x)
    H = fft(coeff)
    Y = X / H
    y = ifft(Y)
    return np.real(y)
