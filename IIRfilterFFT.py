import numpy as np
from scipy.fft import fft, ifft


def IIRfilterFFT(coeff, x):
    # zero pad coeff to match x
    coeff = np.pad(coeff, (0, len(x) - len(coeff)), mode='constant')
    fftInp = fft(x)
    fftCoeff = fft(coeff)
    filtered = fftInp / fftCoeff
    return np.real(ifft(filtered))
