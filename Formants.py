import numpy as np
import sounddevice
from numpy import size
import librosa

# def Formants(input_):
input_path = 'audio/shortbikes.wav'
input_, SR = librosa.load(input_path, mono=True)

analysisHop = 256
synthesisHop = 256
LEN = 2048
hannWin = np.hanning(LEN)
# init
psi = np.zeros((LEN, 1))
ramp = np.floor((np.arange(LEN)) * analysisHop / synthesisHop).astype(int) + 1
previousPhi = np.zeros((LEN, 1))
resampledLEN = np.floor(LEN * analysisHop / synthesisHop).astype(int)
x = 1 + (np.arange(resampledLEN)) * LEN / resampledLEN
x = x.reshape(-1, 1)

input_ = input_ / np.max(np.abs(input_))

output = np.zeros((len(input_) + resampledLEN, 1))
omega = 2 * np.pi * analysisHop * (np.arange(LEN)) / LEN
omega = omega.reshape(-1, 1)

for anCycle in np.arange(0, np.floor(len(input_) / 2 - max(LEN, ramp[-1])) + 1, analysisHop):
    anCycle = anCycle.astype(int)
    grain = input_[anCycle:anCycle + LEN] * hannWin
    grain = grain.reshape(-1, 1)
    fftGrain = np.fft.fft(grain)
    # phase information: output psi
    phi = np.angle(fftGrain)
    delta = np.mod(phi - previousPhi - omega + np.pi, -2 * np.pi) + omega + np.pi
    psi = np.mod(psi + delta * synthesisHop / analysisHop + np.pi, -2 * np.pi) + np.pi
    previousPhi = phi
    # shifting: output correction factor
    f1 = np.abs(np.fft.fft(input_[anCycle + ramp] * hannWin) / (LEN * 0.5))
    f1 = f1.reshape(-1, 1)
    logarithmic = (np.log(0.00001 + f1) - np.log(0.00001 + np.abs(fftGrain))) / 2
    realLog = np.fft.ifft(logarithmic)
    corrected = (np.abs(fftGrain) * np.exp(realLog[0, 0]) * np.exp(1j * psi))
    # interpolation
    grain = (np.real(np.fft.ifft(corrected.T)) * hannWin).T
    temp = np.zeros(np.size(x)).reshape(-1, 1)
    for i in range(len(x) - 1):
        index = np.floor(x[i, 0]).astype(int)
        temp[i] = 2 * grain[index] * x[i]
    output[anCycle:anCycle + resampledLEN] += temp
    print(anCycle)

sounddevice.play(output, SR)
sounddevice.wait()
    # return output
