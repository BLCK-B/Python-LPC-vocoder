def autocorr(input, modelOrder):
    windowSize = len(input)
    denominator = 0.0
    mean = 0.0
    for x in range(windowSize):
        mean += input[x]
    mean /= windowSize
    for sample in input:
        denominator += (sample - mean) ** 2
    R = []
    for k in range(modelOrder + 1):
        numerator = 0.0
        for n in range(windowSize - k):
            sample = input[n]
            lagSample = input[n + k]
            numerator += (sample - mean) * (lagSample - mean)

        R.append(numerator / denominator)

    return R
