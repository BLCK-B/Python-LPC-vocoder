import queue
import threading
import time
import librosa
import numpy as np
import simpleaudio as sa
from matplotlib import pyplot as plt

import graphing
from LPCfun import LPCfun
from LPCfunOptimized import LPCfunOptimized
from graphing import graphqueues
from myFilterIIR import myFilterIIR
from myFFTfilterIIR import myFFTfilterIIR


def play_audio(data, sample_rate):
    # normalize to 16-bit range
    data = data * 32767 / np.max(np.abs(data))
    # convert to 16-bit
    data = data.astype(np.int16)
    play_obj = sa.play_buffer(data, 1, 2, sample_rate)
    play_obj.wait_done()


input_path = 'audio/bikes.wav'
inputc_path = 'audio/raininstr.wav'

voiceAudio, fsVoice = librosa.load(input_path)
carrierAudio, fsCarrier = librosa.load(inputc_path)
print("sample rate: ", fsVoice)
# match sample rate
if fsVoice != fsCarrier:
    carrierAudio = librosa.resample(carrierAudio, fsCarrier, fsVoice)

windowSize = 4500
overlap = 0.5
overlapSize = int(windowSize * overlap)
hopSize = windowSize - overlapSize
windowCount = int(np.floor((len(voiceAudio) - overlapSize) / hopSize))

hann = np.hanning(windowSize)
carrier = np.zeros(windowSize,)
windowTime = windowSize / fsVoice - overlapSize / fsVoice

count = 0
for i in range(windowCount):
    startTime = time.time()

    startIndex = i * hopSize
    endIndex = startIndex + windowSize
    inp = voiceAudio[startIndex:endIndex]
    inpc = carrierAudio[startIndex:endIndex]

    # voice LPC
    inp = hann * inp
    LPC, _ = LPCfunOptimized(inp, 70, False)

    # carrier LPC
    _, e = LPCfunOptimized(inpc, 70, True)
    carrier[1:] = e

    # filter
    output = hann * myFFTfilterIIR(LPC, carrier)
    # normalize
    output = 0.9 * output / np.max(np.abs(output))

    t1 = threading.Thread(target=play_audio, args=(output, fsVoice))
    t1.start()

    # graphing.b1.extend(inp)
    # graphing.b2.extend(output)
    # if count == 5:
    #     graphqueues("raw")
    #     graphing.b1 = []
    #     graphing.b2 = []
    #     count = 0
    # else:
    #     count += 1

    stopTime = time.time()
    timeTaken = stopTime - startTime
    if windowTime > timeTaken:
        time.sleep(windowTime - timeTaken)
    else:
        print("latency overdue by: ", round(timeTaken - windowTime, 4), " s")
