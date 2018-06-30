import numpy as np
from PIL import Image
from Utils import *
from Quantization import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from ImageOperations import *
import matplotlib
from Compression import *

img_name = 'cat'
#img_name = 'lena'

#img_name_in = img_name+'.png'
img_name_in = img_name+'.jpg'

channels = 3
img = open_image(img_name_in, channels)
img = pil_to_np(img)


#Test Cosine

img_out = cosine_transform(img)
img_out2 = np.uint8(img_out)
save_image(img_out2, img_name+' - Cosine_transform_out.png')

img_out = inverse_cosine_transform(img_out)
save_image(img_out, img_name+' - Inverse_cosine_transform_out.png')

mean_square_error(img, img_out, img.shape[2])
signal_to_noise_ration(img, img_out, img.shape[2])

'''
#Test filters

gauss_filter = np.array(gaussian_filter_generator(shape=(9,9),sigma=1));
fimg = convolve(img, gauss_filter, boundary='symm', channels = channels);
save_image(fimg, "wiki_gauss_filter.png")

box_filter = box_filter_generator((9,9))
fimg = convolve(img, box_filter, boundary='symm', channels = channels);
save_image(fimg, "wiki_box_filter.png")

fimg = high_pass_filter(img, selected=0,channels=channels)
save_image(fimg, "wiki_prewitt_filter.png")
#save_image(fimg[..., 1], "wiki_prewitt_filter1.png")
#save_image(fimg[..., 2], "wiki_prewitt_filter2.png")

fimg = high_pass_filter(img, selected=1,channels=channels)
save_image(fimg, "wiki_sobel_filter.png")

fimg = high_pass_filter(img, selected=2,channels=channels)
save_image(fimg, "wiki_roberts_filter.png")

fimg = high_pass_filter(img, selected=3,channels=channels)
save_image(fimg, "wiki_laplacian_filter.png")
'''

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
