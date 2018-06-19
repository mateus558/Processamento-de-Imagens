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
        return;
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