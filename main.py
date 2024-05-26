import threading
import time

import librosa
import numpy as np
import simpleaudio as sa

from LPCfunOptimized import LPCfunOptimized
from myFFTfilterIIR import myFFTfilterIIR


def play_stereo(data, sample_rate):
    if data.shape[0] == 2 and data.shape[1] > data.shape[0]:
        data = data.T
    # normalize to 16-bit range
    data = data * 32767 / np.max(np.abs(data))
    # convert to 16-bit
    data = data.astype(np.int16)
    # ensure the data array is C-contiguous
    data = data.copy(order='C')
    play_obj = sa.play_buffer(data, 2, 2, sample_rate)
    play_obj.wait_done()


def process_channel(inp, inpc, hannw, p):
    # voice LPC
    inp = hannw * inp
    LPC, _ = LPCfunOptimized(inp, p, False)
    # carrier LPC
    _, e = LPCfunOptimized(inpc, p, True)
    # filter
    outputCh = hannw[1:] * myFFTfilterIIR(LPC, e)
    # normalize
    outputCh = 0.9 * outputCh / np.max(np.abs(outputCh))
    return outputCh


input_path = 'audio/anthr.wav'
inputc_path = 'audio/test4.wav'

sampleRate = None
p = 75

voiceAudio, fsVoice = librosa.load(input_path, mono=False, sr=sampleRate)
carrierAudio, _ = librosa.load(inputc_path, mono=False, sr=fsVoice)

if sampleRate is None:
    sampleRate = fsVoice

windowSize = round(0.13 * sampleRate / 100) * 100
print("sample rate: ", fsVoice, "    window size: ", windowSize)
overlap = 0.5
overlapSize = int(windowSize * overlap)
hopSize = windowSize - overlapSize

hann = np.hanning(windowSize)

output = np.zeros((windowSize, 2))
windowTime = windowSize / fsVoice - overlapSize / fsVoice

count = 0
for i in range(10000):
    startTime = time.time()

    startIndex = i * hopSize
    endIndex = startIndex + windowSize

    inpL = voiceAudio[0, startIndex:endIndex]
    inpR = voiceAudio[1, startIndex:endIndex]
    inpcL = carrierAudio[0, startIndex:endIndex]
    inpcR = carrierAudio[1, startIndex:endIndex]

    outputL = process_channel(inpL, inpcL, hann, p)
    outputR = process_channel(inpR, inpcR, hann, p)
    output = np.vstack((outputL, outputR))

    t1 = threading.Thread(target=play_stereo, args=(output, fsVoice))
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
