import numpy as np
from scipy.fft import fft, ifft


def myFFTfilterIIR(coeff, x):
    coeff = np.pad(coeff, (0, len(x) - len(coeff)), mode='constant')
    X = fft(x)
    H = fft(coeff)

    Y = X / H

    y = ifft(Y)
    return np.real(y)


coeff = [1.0, -0.2368, 0.4725, 0.1604, -0.0292, 0.0993, 0.2139]
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]

y = np.zeros_like(x)

for n in range(len(x)):
    y[n] = x[n]
    for k in range(1, len(coeff)):
        if n - k + 1 > 0:
            y[n] -= coeff[k] * y[n - k]

    y[n] /= coeff[0]

for num in y:
    print('{:.4f}'.format(num))

print("FFT: ")
y = myFFTfilterIIR(coeff, x)
for num in y:
    print('{:.4f}'.format(num))
