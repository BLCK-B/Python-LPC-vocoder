import numpy as np

from FFTautocorr import FFTautocorr
from autocorr import autocorr


def LPCfun(inp, p):
    # R = autocorr(inp, p)
    R = FFTautocorr(inp)
    R[0] = 1

    a = np.eye(p + 1, p + 1)
    k = np.zeros((p,))
    E = np.zeros((p,))

    k[0] = -R[1] / R[0]
    a[1, 0] = k[0]
    E[0] = (1 - k[0] ** 2) * R[0]

    # iterations for i > 1
    for i in range(1, p):
        sum_ = 0
        for j in range(i + 1):
            sum_ += a[i, j] * R[j + 1]

        k[i] = -sum_ / E[i - 1]
        a[i + 1, 0] = k[i]

        for j in range(1, i + 1):
            a[i + 1, j] = a[i, j - 1] + k[i] * a[i, i - j]

        E[i] = (1 - k[i] ** 2) * E[i - 1]

    LPC = np.zeros((p + 1,))
    for x in range(p + 1):
        LPC[x] = a[p, p - x]

    LPC[0] = 1.0
    LPC = -LPC

    e = np.zeros(len(inp) - 1)
    for n in range(1, len(inp)):
        sum_ = 0

        for i in range(min(n, p) - 1):
            sum_ += LPC[i + 1] * inp[n - i - 1]

        e[n - 1] = inp[n] - sum_

    return LPC, e