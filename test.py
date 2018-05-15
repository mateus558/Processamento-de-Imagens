import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *

#img_pil = open_image('lena.jpg', channels)
#img = mediancut_algorithm(img_pil, 16, channels)
#img = pil_to_np(img_pil)

#show_image(img, channels)

img_name_in = 'lena.jpg'
img_name_in2 = '255.jpg'
img_name_out = 'lena_gaussian_filter_5.jpg'
channels = 3

r_bits = 3
g_bits = 2
b_bits = 2

img = open_image(img_name_in, channels)

img = pil_to_np(img)

A = np.asarray(img).reshape(-1)
#np.savetxt('text2.txt', A, fmt='%d')

#save_image(img_out, img_name_out)

#show_image_np(img_out, 3)
#if img_out.dtype != np.int:
#    print(img_out[0, :4])
#    #save_image(img_out, img_name_out)
#    show_image_np(img_out, 3)

