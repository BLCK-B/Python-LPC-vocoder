import numpy as np


def LPCfun(inp, p):
    R = np.correlate(inp, inp, mode='full')[:len(inp)]
    R = R * 40

    a = np.eye(p + 1, p + 1)
    k = np.zeros((p,))
    E = np.zeros((p,))
    LPC = np.zeros((p, p))

    k[0] = -R[1] / R[0]
    a[1, 0] = k[0]
    E[0] = (1 - k[0] ** 2) * R[0]

    # Iterations for i > 1
    for i in range(1, p):
        sum_ = 0
        for j in range(i):
            sum_ += a[i, j] * R[j + 1]

        k[i] = -sum_ / E[i - 1]
        a[i + 1, 0] = k[i]

        for j in range(1, i):
            a[i + 1, j] = a[i, j - 1] + k[i] * a[i, i - j]

        E[i] = (1 - k[i] ** 2) * E[i - 1]

    for x in range(p):
        LPC[x] = a[p, p - x]

    LPC[0] = 1.0
    LPC = -LPC

    e = np.zeros(len(inp) - 1)
    for n in range(1, len(inp)):
        sum_ = 0

        for i in range(min(n, p)):
            sum_ += LPC[i + 1] * inp[n - i - 1]

        e[n - 1] = inp[n] - sum_

    return LPC, e
