import numpy as np
from scipy.fft import fft, ifft


def myFFTfilterIIR(coeff, x):
    # Ensure the length of the coefficient matches the expected length after padding
    coeff = np.pad(coeff, (0, len(x) - len(coeff)), mode='constant')
    # Perform FFT on the input signal and the filter coefficients
    X = fft(x)
    H = fft(coeff)
    # Multiply the FFTs in the frequency domain
    Y = X / H
    # Perform inverse FFT to get the filtered signal in the time domain
    y = ifft(Y)
    # Return the real part of the result, as complex numbers may have small imaginary components
    return np.real(y)
