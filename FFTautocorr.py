import numpy as np
from scipy.fftpack import fft, ifft


def FFTautocorr(x):
    psd = np.abs(fft(x)) ** 2
    acorr = ifft(psd).real
    acorr /= acorr[0]
    return acorr
