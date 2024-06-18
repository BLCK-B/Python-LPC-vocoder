import time

import numpy as np
from scipy import signal
from FFTautocorr import FFTautocorr


def LPCfunOptimized(inp, p, errors):
    # R = autocorr(inp, p)

    R = FFTautocorr(inp)
    R[0] = 1

    a = np.eye(p + 1, p + 1)
    k = np.zeros((p,))
    E = np.zeros((p,))

    a[1, 0] = k[0] = -R[1] / R[0]
    E[0] = (1 - k[0] ** 2) * R[0]

    # iterations for i > 1
    for i in range(1, p):
        sum_ = 0
        for j in range(i + 1):
            sum_ += a[i, j] * R[j + 1]

        a[i + 1, 0] = k[i] = -sum_ / E[i - 1]

        for j in range(1, i + 1):
            a[i + 1, j] = a[i, j - 1] + k[i] * a[i, i - j]

        E[i] = (1 - k[i] ** 2) * E[i - 1]

    LPC = np.zeros((p + 1,))
    for x in range(p + 1):
        LPC[x] = a[p, p - x]

    LPC[0] = 1.0

    e = np.zeros(len(inp))

    if errors:
        e = signal.fftconvolve(inp, LPC, mode='same')

    return LPC, e

    # start_time = time.time()
    # print((time.time() - start_time) * 1000)
