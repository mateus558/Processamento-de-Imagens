import numpy as np
import matplotlib.pyplot as plt


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

    if channels == 1:
        hist = compute_histogram(image, 256, 1)
        pmf = hist/size #probability mass function
        cdf = pmf   #cumulative distributive function

        for i in range(1, len(cdf)):
            cdf[i] = cdf[i-1] + cdf[i]
        cdf = cdf * 255
        image[:, :] = cdf[image[:, :]]

        return image
    elif channels == 3:
        r_hist, g_hist, b_hist = compute_histogram(image, 256, 3, False)
        hist = np.array([r_hist, g_hist, b_hist])
        pmf = hist / size
        cdf = pmf

        for j in range(0, cdf.shape[0]):
            for i in range(1, len(cdf)):
                cdf[j][i] = cdf[j][i-1] + cdf[j][i]
        cdf = cdf * 255

        for i in range(0, image.shape[2]):
            for j in range(0, image.shape[0]):
                for k in range(0, image.shape[1]):
                    image[j, k, i] = cdf[i, image[j, k, i]]

        return image