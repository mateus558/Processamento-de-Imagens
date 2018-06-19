import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *


#Test Fourier

img_name_in = 'cat.jpg'
channels = 3

img = open_image(img_name_in, channels)

img = pil_to_np(img)
for i in range(10):
    img_s = resize(img, 1-i*0.1, 3, type="area", kind="slinear")
    show_image_np(img_s, 3)
#fourier_transform_scipy(img)

#fourier_transform(img, 2, 20, 30)


