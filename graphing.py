import queue

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram

b1 = []
b2 = []


def graphqueues(graphType="raw"):
    global b1
    global b2
    plt.close()

    if graphType == "raw":
        fig, (g1, g2) = plt.subplots(1, 2, figsize=(12, 6))
        g1.plot(b1)
        g1.set_title('original file')
        g2.plot(b2)
        g2.set_title('synthesised')

    if graphType == "fft":
        b1 = np.abs(fft(b1))
        b2 = np.abs(fft(b2))
        b1 = b1[0:round(len(b1)/2)]
        b2 = b2[0:round(len(b2)/2)]
        fig, (g1, g2) = plt.subplots(1, 2, figsize=(12, 6))
        g1.plot(b1)
        g1.set_title('original file')
        g2.plot(b2)
        g2.set_title('synthesised')

    if graphType == "sg":
        b1 = np.array(b1)
        freq, t, s = spectrogram(b1, 22050)
        colors = plt.cm.get_cmap('viridis', 256)
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, freq, s, cmap=colors, shading='auto')
        plt.colorbar(label='intensity')
        plt.xlabel('time')
        plt.ylabel('frequency')

    plt.show()

