import numpy as np
import sounddevice
from scipy.fft import fft, ifft
import librosa

# def Formants(input_):
input_path = 'audio/shortbikes.wav'
input_, SR = librosa.load(input_path, mono=True)

analysisHop = 150
synthesisHop = 256
LEN = 2048
hannWin = np.hanning(LEN)
# init
psi = np.zeros((LEN, 1))
ramp = np.floor((np.arange(LEN)) * analysisHop / synthesisHop).astype(int) + 1
previousPhi = np.zeros((LEN, 1))
resampledLEN = np.floor(LEN * analysisHop / synthesisHop).astype(int)
x = 1 + (np.arange(resampledLEN)) * LEN / resampledLEN
x = x.reshape(-1, 1).astype(int)

input_ = input_ / np.max(np.abs(input_))

output = np.zeros((len(input_) + resampledLEN, 1))
omega = 2 * np.pi * analysisHop * (np.arange(LEN)) / LEN
omega = omega.reshape(-1, 1)

endCycle = np.floor(len(input_) - max(LEN, ramp[LEN-1])).astype(int)
for anCycle in np.arange(0, endCycle, analysisHop):
    anCycle = anCycle.astype(int)
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
    temp = np.zeros(np.size(x)).reshape(-1, 1)
    for i in range(len(x)):
        index = np.floor(x[i]).astype(int)
        temp[i] = 2 * grain[index-1] * x[i, 0]
    output[anCycle:anCycle + resampledLEN] += temp

output *= 0.005
sounddevice.play(output, SR)
sounddevice.wait()
    # return output
