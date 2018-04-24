import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(img_np, gamma, channels):
    img_out = img_np
    gamma_correction = 1 / gamma

    height = np.size(img_np, 0)
    width = np.size(img_np, 1)

    for i in range(height):
        for j in range(width):
            if channels == 1:
                new = int( 255 * (img_np[i][j] / 255) ** gamma_correction )
                img_out[i, j] = new
            elif channels == 3:
                new_r = int( 255 * (img_np[i][j][0] / 255) ** gamma_correction )
                new_g = int( 255 * (img_np[i][j][1] / 255) ** gamma_correction )
                new_b = int( 255 * (img_np[i][j][2] / 255) ** gamma_correction )
                img_out[i, j] = np.array([new_r, new_g, new_b])

    return img_out


def compute_histogram(image, bins, channels=1, plot=False):
    interval_size = 256 / bins
    if channels == 1:
        flatten = image.ravel()
        hist = np.zeros(shape=(bins, ), dtype=np.uint64)

        for i in range(0, len(flatten)):
            hist[int(np.floor(flatten[i]/interval_size))] += 1

        if plot:
            plt.bar(np.arange(bins)*interval_size, hist)
            plt.show()

        return hist
    else:
        r = image[:, :, 0].ravel()
        g = image[:, :, 1].ravel()
        b = image[:, :, 2].ravel()
        r_h = np.zeros(shape=(bins,), dtype=np.uint64)
        g_h = np.zeros(shape=(bins,), dtype=np.uint64)
        b_h = np.zeros(shape=(bins,), dtype=np.uint64)
        for i in range(0, len(r)):
            r_h[int(np.floor(r[i] / interval_size))] += 1
            g_h[int(np.floor(g[i] / interval_size))] += 1
            b_h[int(np.floor(b[i] / interval_size))] += 1

        if plot:
            a = plt.subplot(111)
            a.plot(np.arange(bins)*interval_size, r_h, color='r')
            a.plot(np.arange(bins)*interval_size, g_h, color='g')
            a.plot(np.arange(bins)*interval_size, b_h, color='b')
            plt.show()
        return np.array([r_h, g_h, b_h])


def equalize_histogram(image, channels=1):
    size = image.shape[0] * image.shape[1]

    hist = compute_histogram(image, 256, channels)
    hist = sum(hist[:]) / channels

    pmf = hist / size  # probability mass function
    cdf = pmf.cumsum()
    cdf = 255*cdf/cdf[-1]

    image[:, :] = cdf[image[:, :]]

    return image


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
