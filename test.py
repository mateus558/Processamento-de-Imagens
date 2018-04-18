import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *

channels = 1

img_pil = open_image('lena.jpg', channels)
#img = mediancut_algorithm(img_pil, 16, channels)
img = pil_to_np(img_pil)

show_image(img, channels)