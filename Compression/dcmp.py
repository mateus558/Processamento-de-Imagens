import numpy as np

def DCPM(input):
    output = np.zeros(len(input), dtype=int)
    k = 0

    output[k] = input[k]; k += 1

    while(k < len(input)):
        output[k] = input[k] - input[k-1]; k += 1

    return output

def inverse_DCPM(input):
    output = np.zeros(len(input))
    k = 0

    output[k] = input[k]; k += 1

    while(k < len(input)):
        output[k] = output[k-1] + input[k]; k += 1

    return output