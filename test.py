import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *


img_name = 'cat'
img_name = 'wiki'

#img_name_in = img_name+'.jpg'
img_name_in = img_name+'.png'

channels = 3
img = open_image(img_name_in, channels)
img = pil_to_np(img)

gauss_filter = gaussian_filter_generator();
fimg = convolve(gauss_filter, img);

show_image_np(fimg, 3);

'''
#Test Fourier
fourier_transform_scipy(img, 3, 5, 30, img_name)
fourier_transform(img, 3, 5, 30, img_name)
'''

'''
#Test Resize and effects 
img_s = resize(img, 1.0, 3, type='pontual',       kind='linear', img_name='Resize-0%-Original.png')
fourier_transform(img_s, 0, img_name='Resize-0%-Original')
img_s = resize(img, 0.5, 3, type='area',          kind='linear', img_name='Resize-50%-Area.png')
fourier_transform(img_s, 0, img_name='Resize-50%-Area')
img_s = resize(img, 0.5, 3, type='pontual',       kind='linear', img_name='Resize-50%-Pontual.png')
fourier_transform(img_s, 0, img_name='Resize-50%-Pontual')
img_s = resize(img, 1.5, 3, type='interpolation', kind='cubic',     img_name='Resize-150%-Interpolation-Cubic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Cubic')
img_s = resize(img, 1.5, 3, type='interpolation', kind='linear',    img_name='Resize-150%-Interpolation-Linear.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Linear')
img_s = resize(img, 1.5, 3, type='interpolation', kind='quadratic', img_name='Resize-150%-Interpolation-Quadratic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Quadratic')
img_s = resize(img, 1.5, 3, type='nearest',       kind='linear', img_name='Resize-150%-Nearest.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Nearest')
'''