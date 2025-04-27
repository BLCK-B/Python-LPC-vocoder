import numpy as np


def IIRfilter(coeff, x):
    y = np.zeros_like(x)

    for n in range(len(x)):
        y[n] = x[n]
        for k in range(1, len(coeff)):
            if n - k + 1 > 0:
                y[n] -= coeff[k] * y[n - k]

        y[n] /= coeff[0]

    return y
