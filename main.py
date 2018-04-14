import os, sys
from sys import platform
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image

pathname = os.path.dirname(sys.argv[0])

# conversor_image('lena.jpg', 1)
def conversor_image(img_name, channels):
    img = Image.open(os.path.join(pathname, img_name))

    if channels == 1:
        return img.convert('L')
    elif channels == 2:
        return img.convert('LA')
    elif channels == 2:
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


"""
def clear():
    if platform == "linux" or platform == "linux2":
        os.system('clear')
    elif platform == "darwin":
        os.system('clear')
    elif platform == "win32":
        os.system('cls')

def put_alpha_channel(img):
    img.putalpha(1)
    
print ('sys.argv[0] =', sys.argv[0])
print ('path =', pathname)
print ('full path =', os.path.abspath(pathname))

img = Image.open(os.path.join(pathname, 'lena.jpg'))
img_np = np.asarray(img, dtype = np.uint8)

img = conversor_image('lena.jpg', 1)
save_image(img, 'lena_LA_converted.png')

height = np.size(img_np, 0)
width = np.size(img_np, 1)

for i in range(height):
    for j in range(width):
        new_r = int( 0 )
        new_g = int( 0 )
        new_b = int( 0 )
        img.putpixel((j,i),(new_r, new_g, new_b))

img.save(os.path.join(pathname, ('preto.jpg')))

for i in range(height):
    for j in range(width):
        new_r = int( 255 )
        new_g = int( 255 )
        new_b = int( 255 )
        img.putpixel((j,i),(new_r, new_g, new_b))

img.save(os.path.join(pathname, ('branco.jpg')))
"""

#correction_gamma('lena_LA_converted.png', 0.5, 'TC')
#print ( signal_to_noise_ration('correction_gamma_0.5_lena.jpg', 'correction_gamma_0.5_lena.jpg', 'RGB') )
