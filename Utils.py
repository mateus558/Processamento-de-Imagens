import sys, os
import numpy as np
from PIL import Image

pathname = os.path.dirname(sys.argv[0])

pathname_image_in = pathname+'images/'
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


def show_image(img):
    img.show()


def save_image(img, img_name):
    img.save(os.path.join(pathname_image_out, img_name))


def np_to_pil(image):
    return Image.fromarray(image)


def pil_to_np(image):
    img = np.asarray(image, dtype = np.uint8)
    img.setflags(write=1)
    return img