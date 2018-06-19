import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *


img_name1 = 'cat'
img_name = 'lena'

#img_name_in = img_name+'.jpg'
img_name_in = img_name+'.jpg'

channels = 1
img = open_image(img_name_in, channels)
img = pil_to_np(img)

gauss_filter = np.array(gaussian_filter_generator(shape=(9,9),sigma=1));
fimg = convolve(img, gauss_filter, boundary='symm', channels = channels);
save_image(fimg, "wiki_gauss_filter_gray.png")

box_filter = box_filter_generator((9,9))
fimg = convolve(img, box_filter, boundary='symm', channels = channels);
save_image(fimg, "wiki_box_filter_gray.png")

fimg = high_pass_filter(img, selected=0,channels=channels)
save_image(fimg, "wiki_prewitt_filter_gray.png")

fimg = high_pass_filter(img, selected=1,channels=channels)
save_image(fimg, "wiki_sobel_filter_gray.png")

fimg = high_pass_filter(img, selected=2,channels=channels)
save_image(fimg, "wiki_roberts_filter_gray.png")

fimg = high_pass_filter(img, selected=3,channels=channels)
save_image(fimg, "wiki_laplacian_filter_gray.png")


'''
#Test Fourier

fourier_transform_scipy(img, 3, 5, 30, img_name)
fourier_transform(img, 3, 5, 30, img_name)


#Test Resize and effects 

img_s = resize(img, 1.0, 3, type='pontual',       kind='linear', img_name='Resize-0%-Original.png')
fourier_transform(img_s, 0, img_name='Resize-0%-Original')

img_s = resize(img, 0.5, 3, type='area',          kind='linear', img_name='Resize-50%-Area.png')
fourier_transform(img_s, 0, img_name='Resize-50%-Area')

img_s = resize(img, 0.5, 3, type='pontual',       kind='linear', img_name='Resize-50%-Pontual.png')
fourier_transform(img_s, 0, img_name='Resize-50%-Pontual')


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

img_s = resize(img, 1.5, 3, type='interpolation', kind='cubic',     img_name='Resize-150%-Interpolation-Cubic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Cubic')

img_s = resize(img, 1.5, 3, type='interpolation', kind='linear',    img_name='Resize-150%-Interpolation-Linear.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Linear')

img_s = resize(img, 1.5, 3, type='interpolation', kind='quadratic', img_name='Resize-150%-Interpolation-Quadratic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Quadratic')

img_s = resize(img, 1.5, 3, type='nearest',       kind='linear', img_name='Resize-150%-Nearest.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Nearest')
'''
