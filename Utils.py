import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


pathname = os.path.dirname(sys.argv[0])


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


def show_image(img, channels):
    if channels == 1:
        plt.imshow(img, cmap='gray')
    else: plt.imshow(img)
    plt.show()


# save_image(img, 'lena.jpg')
def save_image(img, img_name):
    img.save(os.path.join(pathname, img_name))


def np_to_pil(image):
    return Image.fromarray(image)


def pil_to_np(image):
    img = np.asarray(image, dtype = np.uint8)
    img.setflags(write=1)
    return img


# correction_gamma('lena.jpg', 0.5, 'RBG')
def correction_gamma(img_np, gamma, channels):
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