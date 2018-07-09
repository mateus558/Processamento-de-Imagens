import sys, os
import numpy as np
from PIL import Image
#from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import scipy.misc

pathname = os.path.dirname(os.path.abspath(__file__))

pathname_image_in = pathname + '/images/'
pathname_image_out = pathname_image_in+'out/'

if not os.path.exists(pathname_image_in):
    os.makedirs(pathname_image_in)
if not os.path.exists(pathname_image_out):
    os.makedirs(pathname_image_out)


def rgb_to_ycbcr(img):
    xform = np.array([[ 0.299,   0.587,   0.114],
                      [-0.1687, -0.3313,  0.5],
                      [ 0.5,    -0.4187, -0.0813]])

    ycbcr = img.dot(xform.T)
    ycbcr[:, :, [1,2]] += 128.0
    
    return ycbcr


def ycbcr_to_rgb(img):
    xform = np.array([[1.0,  0.0,      1.402],
                      [1.0, -0.34414, -0.71414],
                      [1.0,  1.772,    0.0]])

    rgb = img.astype(np.float)
    rgb[:, :, [1,2]] -= 128.0

    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0,   0)

    return np.uint8(rgb)


def open_image(img_name, channels):
    img = Image.open(os.path.join(pathname_image_in, img_name))

    if channels == 1:
        return img.convert('L')
    elif channels == 2:
        return img.convert('LA')
    elif channels == 3:
        return img.convert('RGB')
    else:
        return img.convert('RGBA')


def show_image_PIL(img):
    img.show()


def show_image_np(img, channels):
    '''
    if img.dtype == np.uint16:
        viewer = ImageViewer(img)
        viewer.show()
        skimage.imshow(img)
        return
    '''
    if channels == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

def save_image(img, img_name='out.png'):
    img = np_to_pil(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(os.path.join(pathname_image_out, img_name))


def np_to_pil(image):
    return Image.fromarray(image)


def pil_to_np(image):
    img = np.asarray(image, dtype = np.uint8)
    img.setflags(write=1)
    return img

def get_path_name():
    return pathname