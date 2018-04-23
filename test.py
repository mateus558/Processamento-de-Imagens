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

img_name_in = 'lena.jpg'
img_name_in2 = '255.jpg'
img_name_out = 'lena_quantization_10_10_10.jpg'
channels = 3

r_bits = 10
g_bits = 10
b_bits = 10

img = open_image(img_name_in, channels)

img_out = quantization(img, channels, r_bits, g_bits, b_bits)

img_out = np_to_pil(img_out)

save_image(img_out, img_name_out)
show_image(img_out)