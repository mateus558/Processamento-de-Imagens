import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *

#img_pil = open_image('lena.jpg', channels)
#img = mediancut_algorithm(img_pil, 16, channels)
#img = pil_to_np(img_pil)

#show_image(img, channels)

img_name_in = 'lena_eq.jpg'
img_name_in2 = '255.jpg'
img_name_out = 'lena_quantization_10_10_10.jpg'
channels = 3

r_bits = 2
g_bits = 2
b_bits = 2

img = open_image(img_name_in, channels)
img = pil_to_np(img)
img_out = quantization(img, channels, r_bits, g_bits, b_bits)

show_image_np(img_out,3)
#if img_out.dtype != np.int:
#    print(img_out[0, :4])
#    #save_image(img_out, img_name_out)
#    show_image_np(img_out, 3)

