import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *




img_name_in = 'cat.jpg'
channels = 3
img = open_image(img_name_in, channels)
img = pil_to_np(img)


#Test Fourier

fourier_transform_scipy(img, 3, 5, 30)
fourier_transform(img, 3, 5, 30)
'''
#Test Resize

img_s = resize(img, .5, 3, type="pontual", kind="slinear")
'''

