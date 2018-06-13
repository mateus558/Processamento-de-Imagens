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
img_s = resize(img, .5, 3, type="pontual", kind="slinear")

show_image_np(img_s, 3)
#fourier_transform_scipy(img)

#fourier_transform(img, 2, 20, 30)


