import matplotlib.pyplot as plt
from Utils import pil_to_np
import numpy as np
from PIL import Image


def popularity_algorithm(img, ncolors, channels):
    colors = img.getcolors(img.size[0]*img.size[1])
    most_pop = []
    print(len(colors))
    for c in range(ncolors):
        maxcount = -np.inf
        ind = None
        for i in range(len(colors)):
            if colors[i][0] > maxcount:
                maxcount = colors[i][0]
                ind = i
        most_pop.append(colors[ind][1])
        colors.remove(colors[ind])
    print(most_pop)
    print("%d Most frequent colors found." % (ncolors))
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
