from Histogram import compute_histogram
import numpy as np


def global_limiarization(image, threshold):
    bin = np.zeros((image.shape[0], image.shape[1],), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(image[i, j] > threshold):
                bin[i, j] = 255
    return bin


def otsu_thresholding(img):
    m, n = img.shape
    size = m*n
    p = compute_histogram(img, 256, 1)/size
    psum = np.cumsum(p)
    mk = np.zeros((256,), dtype=np.float)
    var = np.zeros((256,), dtype=np.float)

    varg = 0
    mg = 0
    p1 = 0
    p2 = 0
    m1 = 0
    m2 = 0

    for i in range(len(psum)):
        mg += i * p[i]
        mk[i] = mg

    for i in range(len(psum)):
        p1 += p[i]
        p2 = 1 - p1

        if p1 != 0:
            m1 = mk[i] / p1
        if p2 != 0:
            m2 = (mg - mk[i]) / p2
        var[i] = p1*p2*(m1-m2)*(m1-m2)

        varg += (i - mg) * (i - mg) * p[i]

    kstar = np.argmax(var)
    n = var[kstar] / varg
    bin = global_limiarization(img, kstar)

    return bin, kstar, n
