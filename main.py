import threading
import librosa
import numpy as np
import simpleaudio as sa

from LPCfun import LPCfun
from myFilterIIR import myFilterIIR

input_path = 'audio/anthr.wav'
inputc_path = 'audio/mid.wav'


def play_audio(data, sample_rate):
    # normalize to 16-bit range
    data = data * 32767 / np.max(np.abs(data))
    # convert to 16-bit
    data = data.astype(np.int16)
    play_obj = sa.play_buffer(data, 1, 2, sample_rate)
    play_obj.wait_done()


voiceAudio, fsVoice = librosa.load(input_path)
carrierAudio, fsCarrier = librosa.load(inputc_path)
# match sample rate
if fsVoice != fsCarrier:
    carrierAudio = librosa.resample(carrierAudio, fsCarrier, fsVoice)

p = 30

windowSize = 6000
overlap = 0.5
overlapSize = int(windowSize * overlap)
hopSize = windowSize - overlapSize
windowCount = int(np.floor((len(voiceAudio) - overlapSize) / hopSize))

hann = np.hanning(windowSize)

for i in range(windowCount):
    # print(i)
    startIndex = i * hopSize
    endIndex = startIndex + windowSize
    inp = voiceAudio[startIndex:endIndex]
    inpc = carrierAudio[startIndex:endIndex]
    # carrier LPC
    carrier = np.zeros((len(inp),))
    _, e = LPCfun(inpc, 5)
    carrier[1:len(inpc)] = e
    # voice LPC
    inp = hann * inp
    LPC, _ = LPCfun(inp, p)

    # filter
    output = hann * myFilterIIR(LPC, carrier)
    # normalize
    output = 0.9 * output / np.max(np.abs(output))

    t1 = threading.Thread(target=play_audio, args=(output, fsVoice))
    t1.start()

    # pauseDuration = len(output) / fsVoice - overlapSize / fsVoice
    # time.sleep(pauseDuration)
