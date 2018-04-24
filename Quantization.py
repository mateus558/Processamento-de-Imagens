from Utils import *
import numpy as np
from PIL import Image


def popularity_algorithm(img, ncolors, ball_size, channels):
    colors = img.getcolors(img.size[0]*img.size[1])
    most_pop = []
    while len(most_pop) < ncolors:
        maxcount = -np.inf
        ind = None
        ind_warn = None
        warn = False

        for i in range(len(colors)):
            if colors[i][0] > maxcount:
                for pop in range(len(most_pop)):
                    p = np.array(most_pop[pop])
                    a = np.array(colors[i][1])

                    if np.linalg.norm(p - a) < ball_size:
                        warn = True
                        ind_warn = pop
                maxcount = colors[i][0]
                ind = i
        if ind is None:
            ball_size -= 1
            return popularity_algorithm(img, ncolors, ball_size, channels)
        else:
            if warn:
                most_pop.remove(most_pop[ind_warn])
            most_pop.append(colors[ind][1])
            colors.remove(colors[ind])

    print("%d Most frequent colors found. ball_size = %d." % (ncolors, ball_size))
    img = pil_to_np(img)
    if channels == 1: m, n = img.shape
    else: m, n, _ = img.shape

    for i in range(m):
        for j in range(n):
            old_dist = np.inf
            ind = None
            for k in range(ncolors):
                dist = np.linalg.norm(img[i, j] - most_pop[:][k])

                if dist < old_dist:
                    ind = k
                    old_dist = dist
            img[i, j] = most_pop[:][ind]
    return img


def mediancut_algorithm(img, ncolors, channels):
    m = img.size[1]
    n = img.size[0]
    size = m * n
    colors = img.getcolors(size)
    col = []
    upper_pix = np.full((channels, ), -np.inf, dtype=np.uint8)
    lower_pix = np.full((channels, ), np.inf, dtype=np.uint8)
    pallete = []

    for _, c in colors:
        if channels > 1:
            for k in range(channels):
                lower_pix[k] = min(lower_pix[k], c[k])
                upper_pix[k] = max(upper_pix[k], c[k])
        else:
            lower_pix = min(lower_pix, c)
            upper_pix = max(upper_pix, c)
        col.append(c)

    range_pix = upper_pix - lower_pix
    indmaxch = np.argmax(range_pix)

    if channels > 1:
        col.sort(key=lambda x: x[indmaxch])
    else:
        col.sort()

    buckets = []
    buckets.append(col[:int(len(col)/2)])
    buckets.append(col[int(len(col)/2):])
    while len(buckets) < ncolors:
        for i in range(len(buckets)):
            bucket = buckets[i]
            buckets.append(bucket[:int(len(bucket)/2)])
            buckets.append(bucket[int(len(bucket)/2):])
            del buckets[i]

    rounder = lambda vi: round(vi)
    rfunc = np.vectorize(rounder)

    for bucket in buckets:
        sum = np.zeros((channels,))

        for color in bucket:
            sum = sum + np.array(color)
        sum = rfunc(np.array(sum) / len(bucket))
        pallete.append(sum)

    img = pil_to_np(img)
    for i in range(0, m):
        for j in range(0, n):
            old_dist = np.inf
            new_color = None
            for color in pallete:
                dist = np.linalg.norm(img[i, j] - color)
                if dist < old_dist:
                    old_dist = dist
                    new_color = color
            img[i, j] = new_color
    return img

