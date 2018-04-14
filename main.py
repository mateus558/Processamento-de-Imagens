import os, sys
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

pathname = os.path.dirname(sys.argv[0])

def clear():
    if platform == "linux" or platform == "linux2":
        os.system('clear')
    elif platform == "darwin":
        os.system('clear')
    elif platform == "win32":
        os.system('cls')

"""
def put_alpha_channel(img):
    img.putalpha(1)
    
print ('sys.argv[0] =', sys.argv[0])
print ('path =', pathname)
print ('full path =', os.path.abspath(pathname))
"""


def is_grey_scale(image):
    img = image.convert('RGB')
    width, height = img.size
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i,j))
            if r != g != b: return False
    return True


def conversor_image(img):
    if(is_grey_scale(img)):
        return img.convert('RGB')
    else:
        return img.convert('L')


def save_image(img):
    if(img.format == 'JPG'):
        img.save('out','jpg')
    else:
        img.save('out','png')


def squared_error(img1_np, img2_np):
    width = np.size(img1_np,1)
    height = np.size(img1_np,0)

    mean = 0.0
    for i in range(height):
        for j in range(width):
            for k in range(0,3):
                mean += (img1_np[i][j][k] - img2_np[i][j][k]) ** 2

    #for i1 in img1_np and i2 in img2_np:
     #   for j1 in i1 and j2 in i2:
      #      for k1 in j1 and k2 in j2:
       #         mean += (k1 - k2) ** 2

    return mean


def mean_square_error(img1, img2):
    width, height = img1.size
    width2, height2 = img2.size

    if width == width2 and height == height2:
        img1_np = np.asarray(img1, dtype = np.uint8)
        img2_np = np.asarray(img2, dtype = np.uint8)

        return squared_error(img1_np, img2_np) / (width * height)
    else:
        return 0


def signal_to_noise_ration(original_img, noisy_img):
    img1 = Image.open(os.path.join(pathname, original_img))
    img2 = Image.open(os.path.join(pathname, noisy_img))

    original_img_np = np.asarray(img1, dtype = np.uint8)
    noisy_img_np = np.asarray(img2, dtype = np.uint8)

    height = np.size(original_img_np, 0)
    width = np.size(original_img_np, 1)

    height2 = np.size(noisy_img_np, 0)
    width2 = np.size(noisy_img_np, 1)

    if width == width2 and height == height2:
        signal = 0.0
        #for i in range(height):
         #   for j in range(width):
          #      for k in range(0,3):
           #         signal += (noisy_img_np[i][j][k]) ** 2

        for i in noisy_img_np:
            for j in i:
                for k in j:
                    signal += k ** 2

        return signal / squared_error(original_img_np, noisy_img_np)
    else:
        return 0


def correction_gamma(image, gamma):
    img = Image.open(os.path.join(pathname, image))
    img_np = np.asarray(img, dtype = np.uint8)

    gamma_correction = 1 / gamma

    height = np.size(img_np, 0)
    width = np.size(img_np, 1)

    for i in range(height):
        for j in range(width):
            new_r = int( 255 * (img_np[i][j][0] / 255) ** gamma_correction )
            new_g = int( 255 * (img_np[i][j][1] / 255) ** gamma_correction )
            new_b = int( 255 * (img_np[i][j][2] / 255) ** gamma_correction )
            img.putpixel((j,i),(new_r, new_g, new_b))

    img.save(os.path.join(pathname, ('correction_gamma_('+str(gamma)+')_'+image)))
    img.show()


signal_to_noise_ration('teste.jpg', 'correction_gamma_(5.0)_teste.jpg')
