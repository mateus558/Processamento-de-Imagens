from Utils import *
import numpy as np
from PIL import Image


def popularity_algorithm(img, ncolors, ball_size, channels):
    colors = img.getcolors(img.size[0]*img.size[1])
    most_pop = []
    visited = np.zeros((len(colors), ), dtype=np.bool)
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
