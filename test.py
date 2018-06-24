import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *


#img_name = 'test'
img_name = 'cat'

#img_name_in = img_name+'.png'
img_name_in = img_name+'.jpg'

channels = 3
img = open_image(img_name_in, channels)
img = pil_to_np(img)

'''
#Shannon-Whitaker

img_s = resize(img, 0.5, 3, type='pontual', kind='linear', img_name='Resize-50%-Pontual.png')
img_as_array = np.asarray(img_s).reshape(-1)
img_as_array_out = np.uint8(np.fft.fft(img_as_array))

img_np_out = np.zeros((img_s.shape[0], img_s.shape[1], img_s.shape[2]), dtype=np.uint8)

width = img_s.shape[1]

for u in range (img_s.shape[0]):
    for v in range (img_s.shape[1]):
        i = (u*width*3)+(v*3)
        img_np_out[u][v][0] = img_as_array_out[i]
        img_np_out[u][v][1] = img_as_array_out[i+1]
        img_np_out[u][v][2] = img_as_array_out[i+2]

save_image(img_np_out, str(img_name)+'_SW-Resize-50%-Pontual-Resize-Fourier.png')

img_np_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

img_as_array = np.asarray(img).reshape(-1)
img_as_array_out = np.uint8(np.fft.fft(img_as_array))

for u in range (img.shape[0]):
    for v in range (img.shape[1]):
        i = (u*width*3)+(v*3)
        img_np_out[u][v][0] = img_as_array_out[i]
        img_np_out[u][v][1] = img_as_array_out[i+1]
        img_np_out[u][v][2] = img_as_array_out[i+2]

img_out = resize(img_np_out, 0.5, 3, type='pontual', kind='linear', img_name='Resize-50%-Pontual.png')
save_image(img_out, str(img_name)+'_SW-Resize-50%-Pontual-Fourier-Resize.png')
'''

'''
#Test Blur and Highlight

for i in range(0, 11):
	w = -1.0 + 0.2 * i
	dimension = 9
	img_out = blur_and_highlight_filter(img, w, dimension)
	save_image(img_out, 'blur_and_highlight_filter_'+str(w)+'_'+str(dimension)+'.png')
'''


#Test Filters

gauss_filter = np.array(gaussian_filter_generator(shape=(9,9),sigma=1))
fimg = convolve(img, gauss_filter, boundary='symm', channels=channels)
save_image(fimg, img_name+' - gauss_filter.png')

box_filter = box_filter_generator((9,9))
fimg = convolve(img, box_filter, boundary='symm', channels=channels)
save_image(fimg, img_name+' - box_filter.png')

fimg = high_pass_filter(img, selected=0, channels=channels)
save_image(fimg, img_name+' - prewitt_filter.png')

fimg = high_pass_filter(img, selected=1, channels=channels)
save_image(fimg, img_name+' - sobel_filter.png')

fimg = high_pass_filter(img, selected=2, channels=channels)
save_image(fimg, img_name+' - roberts_filter.png')

fimg = high_pass_filter(img, selected=3, channels=channels)
save_image(fimg, img_name+' - laplacian_filter.png')

channels = 1
img = open_image(img_name_in, channels)
img = pil_to_np(img)

fimg = high_pass_filter(img, selected=3,channels=channels)
save_image(fimg, img_name+' - laplacian_filter_gray.png')


'''
#Test Fourier
fourier_transform_scipy(img, 3, 5, 30, img_name)

img_real_ft, img_imag_ft = fourier_transform(img, 3, 5, 30, img_name)
inverse_fourier_transform(img, img_real_ft, img_imag_ft, img_name)
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

img_s = resize(img, 1.5, 3, type='interpolation', kind='cubic',     img_name='Resize-150%-Interpolation-Cubic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Cubic')

img_s = resize(img, 1.5, 3, type='interpolation', kind='linear',    img_name='Resize-150%-Interpolation-Linear.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Linear')

img_s = resize(img, 1.5, 3, type='interpolation', kind='quadratic', img_name='Resize-150%-Interpolation-Quadratic.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Interpolation-Quadratic')

img_s = resize(img, 1.5, 3, type='nearest',       kind='linear', img_name='Resize-150%-Nearest.png')
fourier_transform(img_s, 0, img_name='Resize-150%-Nearest')
'''
