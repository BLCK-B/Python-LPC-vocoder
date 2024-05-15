import numpy as np


def myFilterIIR(coeff, x):
    y = np.zeros_like(x)

    for n in range(len(x)):
        y[n] = x[n]
        for k in range(2, len(coeff)):
            if n - k + 2 > 0:
                y[n] -= coeff[k] * y[n - k + 1]

        y[n] /= coeff[0]

    return y
