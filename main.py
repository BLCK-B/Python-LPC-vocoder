import time

import numpy as np
import librosa
import sounddevice as sd

from LPCfun import LPCfun
from myFilterIIR import myFilterIIR

input_path = 'audio/anthr.wav'
inputc_path = 'audio/offender.wav'

voiceAudio, fsVoice = librosa.load(input_path)
carrierAudio, fsCarrier = librosa.load(inputc_path)
# match sample rate
if fsVoice != fsCarrier:
    carrierAudio = librosa.resample(carrierAudio, fsCarrier, fsVoice)
# mono
# missing

p = 100

windowSize = 6000
overlap = 0.5
overlapSize = int(windowSize * overlap)
hopSize = windowSize - overlapSize
windowCount = int(np.floor((len(voiceAudio) - overlapSize) / hopSize))

hann = np.hanning(windowSize)

for i in range(windowCount):
    startIndex = i * hopSize
    endIndex = startIndex + windowSize
    inp = voiceAudio[startIndex:endIndex]
    inpc = carrierAudio[startIndex:endIndex]

    carrier = np.zeros((len(inp),))

    _, e = LPCfun(inpc, 100)
    carrier[1:] = e

    # get LPC coeffs and error variance
    inp = hann * inp
    LPC, G = LPCfun(inp, p)
    LPC = -LPC[1:]
    LPC = LPC.T

    # filter
    # missing sqrt of error variance
    output = hann / myFilterIIR(LPC, carrier)
    # normalize
    output = 0.9 * output / np.max(np.abs(output))

    sd.play(output, fsVoice)
    # wait for the duration of the output minus the duration of the overlap
    pauseDuration = len(output) / fsVoice - overlapSize / fsVoice
    time.sleep(pauseDuration)