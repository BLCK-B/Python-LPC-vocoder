import numpy as np
from scipy.fft import fft, ifft


def formantShift(input_):
    # analysisHop = 150
    # analysisHop = 250
    # analysisHop = 100
    analysisHop = 200

    synthesisHop = 150
    LEN = 1000
    hannWin = np.hanning(LEN)
    # init
    psi = np.zeros((LEN, 1))
    previousPhi = np.zeros((LEN, 1))
    ramp = np.floor((np.arange(LEN)) * analysisHop / synthesisHop).astype(int) + 1
    resampledLEN = np.floor(LEN * analysisHop / synthesisHop).astype(int)
    x = 1 + (np.arange(resampledLEN)) * LEN / resampledLEN
    x = x.reshape(-1, 1).astype(int)

    input_ = input_ / np.max(np.abs(input_))

    output = np.zeros((len(input_) + resampledLEN, 1))
    omega = 2 * np.pi * analysisHop * (np.arange(LEN)) / LEN
    omega = omega.reshape(-1, 1)

    endCycle = np.floor(len(input_) - max(LEN, ramp[LEN-1])).astype(int)
    for anCycle in np.arange(0, endCycle, analysisHop):
        grain = input_[anCycle:anCycle + LEN] * hannWin
        fftGrain = fft(grain)
        # phase information: output psi
        phi = np.angle(fftGrain)
        delta = np.mod(phi.reshape(-1, 1) - previousPhi - omega + np.pi, -2 * np.pi) + omega + np.pi
        psi = np.mod(psi + delta * synthesisHop / analysisHop + np.pi, -2 * np.pi) + np.pi
        previousPhi = phi.reshape(-1, 1)
        # shifting: output correction factor
        f1 = np.abs(fft(input_[anCycle + ramp - 1] * hannWin) / (LEN * 0.5))
        logarithmic = (np.log(0.00001 + f1) - np.log(0.00001 + np.abs(fftGrain))) / 2
        realLog = ifft(logarithmic)
        corrected = (np.abs(fftGrain).reshape(-1, 1) * np.exp(realLog[0]) * np.exp(1j * psi))
        # interpolation
        grain = (np.real(ifft(corrected.T)) * hannWin).T
        output[anCycle:anCycle + resampledLEN] += (grain[np.floor(x).astype(int) - 1]).reshape(-1, 1)

    return output[:len(input_)].reshape(-1)

