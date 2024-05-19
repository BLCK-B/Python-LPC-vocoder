import threading
import time

import numpy as np
import librosa
import sounddevice as sd

from LPCfun import LPCfun
from myFilterIIR import myFilterIIR


def play_audio(output, fsVoice):
    sd.play(output, fsVoice, blocking=True)


input_path = 'audio/anthr.wav'
inputc_path = 'audio/anthr.wav'

voiceAudio, fsVoice = librosa.load(input_path)
carrierAudio, fsCarrier = librosa.load(inputc_path)

# match sample rate
if fsVoice != fsCarrier:
    carrierAudio = librosa.resample(carrierAudio, fsCarrier, fsVoice)
# mono
voiceAudio = librosa.to_mono(voiceAudio)
carrierAudio = librosa.to_mono(carrierAudio)

p = 150

windowSize = 6000
overlap = 0
overlapSize = int(windowSize * overlap)
hopSize = windowSize - overlapSize
windowCount = int(np.floor((len(voiceAudio) - overlapSize) / hopSize))

hann = np.hanning(windowSize)

for i in range(windowCount):
    print(i)
    startIndex = i * hopSize
    endIndex = startIndex + windowSize
    inp = voiceAudio[startIndex:endIndex]
    inpc = carrierAudio[startIndex:endIndex]
    # carrier LPC
    carrier = np.zeros((len(inp),))
    _, e = LPCfun(inpc, 50)
    carrier[1:len(inpc)] = e
    # voice LPC
    inp = hann * inp
    LPC, _ = LPCfun(inp, p)

    # filter
    output = hann * myFilterIIR(LPC, carrier)
    # normalize
    output = 0.9 * output / np.max(np.abs(output))

    audio_thread = threading.Thread(target=play_audio, args=(output, fsVoice))
    audio_thread.start()

    # pauseDuration = len(output) / fsVoice - overlapSize / fsVoice
    # time.sleep(pauseDuration)
