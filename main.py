import os, sys
from sys import platform
import numpy as np
import math
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image

pathname = os.path.dirname(sys.argv[0])

# conversor_image('lena.jpg', 1)
def open_image(img_name, channels):
    img = Image.open(os.path.join(pathname, img_name))

    if channels == 1:
        return img.convert('L')
    elif channels == 2:
        return img.convert('LA')
    elif channels == 3:
        return img.convert('RGB')
    else:
        return img.convert('RGBA')


# save_image(img, 'lena.jpg')
def save_image(img, img_name):
    img.save(os.path.join(pathname, (img_name)))


# squared_error(img_np, img2_np, 'TC')
def squared_error(img1_np, img2_np, format):
    height = np.size(img1_np,0)
    width = np.size(img1_np,1)

    error = 0.0
    for i in range (height):
        for j in range (width):
            if format == 'TC':
                error += ((img1_np[i][j] - img2_np[i][j]) ** 2)
            if format == 'RGB':
                for k in range (0,3):
                    img1_aux = np.int(img1_np[i][j][k])
                    img2_aux = np.int(img2_np[i][j][k])
                    error += ((img1_aux - img2_aux) ** 2)

    return error


# mean_square_error(img1_np, img2_np, 'TC')
def mean_square_error(img1_name, img2_name, format):
    img1 = Image.open(os.path.join(pathname, img1_name))
    img2 = Image.open(os.path.join(pathname, img2_name))

    img1_np = np.asarray(img1, dtype = np.uint8)
    img2_np = np.asarray(img2, dtype = np.uint8)

    height = np.size(img1_np, 0)
    width = np.size(img1_np, 1)

    height2 = np.size(img2_np, 0)
    width2 = np.size(img2_np, 1)

    if width == width2 and height == height2:
        if format == 'TC':
            return squared_error(img1_np, img2_np, format) / (width * height)
        elif format == 'RGB':
            return squared_error(img1_np, img2_np, format) / (width * height * 3)
    else:
        return 0


# signal_to_noise_ration('lena.jpg', 'lena2.jpg', 'RBG')
def signal_to_noise_ration(original_img_name, noisy_img_name, format):
    img1 = Image.open(os.path.join(pathname, original_img_name))
    img2 = Image.open(os.path.join(pathname, noisy_img_name))

    original_img_np = np.asarray(img1, dtype = np.uint8)
    noisy_img_np = np.asarray(img2, dtype = np.uint8)

    height = np.size(original_img_np, 0)
    width = np.size(original_img_np, 1)

    height2 = np.size(noisy_img_np, 0)
    width2 = np.size(noisy_img_np, 1)

    if width == width2 and height == height2:
        signal = 0.0
        for i in range(height):
            for j in range(width):
                if format == 'TC':
                    signal += (noisy_img_np[i][j]) ** 2
                if format == 'RGB':
                    for k in range (0,3):
                        signal += (noisy_img_np[i][j][k]) ** 2

        return ( signal / squared_error(original_img_np, noisy_img_np, format) )
    else:
        return 0


# correction_gamma('lena.jpg', 0.5, 'RBG')
def correction_gamma(image, gamma, format):
    img = Image.open(os.path.join(pathname, image))
    img_np = np.asarray(img, dtype = np.uint8)

    gamma_correction = 1 / gamma

    height = np.size(img_np, 0)
    width = np.size(img_np, 1)

    for i in range(height):
        for j in range(width):
            if format == 'TC':
                new = int( 255 * (img_np[i][j] / 255) ** gamma_correction )
                img.putpixel((j, i), (new))
            elif format == 'RGB':
                new_r = int( 255 * (img_np[i][j][0] / 255) ** gamma_correction )
                new_g = int( 255 * (img_np[i][j][1] / 255) ** gamma_correction )
                new_b = int( 255 * (img_np[i][j][2] / 255) ** gamma_correction )
                img.putpixel((j,i),(new_r, new_g, new_b))

    img.save(os.path.join(pathname, ('correction_gamma_'+str(gamma)+'_'+image)))
    img.show()


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


def popularity_algorithm(img, ncolors, channels):
    colors = img.getcolors(img.size[0]*img.size[1])
    most_pop = []

    for c in range(ncolors):
        maxcount = -np.inf
        ind = None
        for i in range(len(colors)):
            if colors[i][0] > maxcount:
                maxcount = colors[i][0]
                ind = i
        most_pop.append(colors[i][1])
        colors.remove(colors[i])

    print("Most frequent colors found.")
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
    plt.imshow(img)
    plt.show()
    return img


def np_to_pil(image):
    return Image.fromarray(image)


def pil_to_np(image):
    img = np.asarray(image, dtype = np.uint8)
    img.setflags(write=1)
    return img


img = open_image('lena.jpg', 3)
popularity_algorithm(img, 60, 3)
img = pil_to_np(img)

#compute_histogram(img, 256, 1, True)
#img = equalize_histogram(img, 1)
#bin = global_limiarization(img, 90)
#print(bin)
#bin, _, _ = otsu_thresholding(img)
#plt.imshow(bin, cmap='gray')
#plt.show()
img = np_to_pil(img)
save_image(img, "eq_lena.jpg")
