import subprocess
import Utils
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

#
#	boundary:
#		- 'fill': constante
#		- 'wrap': periódica
#		- 'symm': simétrica
#	
#	fill_value: valor da constante
#
def convolve(img1_np, img2_np, boundary='fill', fill_value=0):
	if len(img1_np.shape) > 2:
		img_out = signal.convolve2d(img1_np, img2_np, boundary, fill_value)
	else:
		img_out = signal.convolve(img1_np, img2_np)

	return img_out

#def low_pass_filter_generator(dimension, degree):

#Tem que passar um canal por vez
def gaussian_filters(img_np, sigma=7):
	return gaussian_filter(img_np, sigma)


def fourier_transform(img_np):
	img_as_array = np.asarray(img_np).reshape(-1)

	program = get_path_name() + '/Fourier-Transform.exe'

	file_name = 'img_array.txt'
	np.savetxt(file_name, img_np, fmt='%d')

	arguments = (str(img_np.shape[0]) + ' ' + str(img_np.shape[1]) + ' ' + str(img_np.shape[2]) + ' ' + file_name)
	print (arguments)
	subprocess.call([program, arguments])



'''
# Call an exe file with additional parameters
program = get_path_name() + '/Fourier-Transform.exe'
arguments = ('argument1 argument2 argument3 argument4')
subprocess.call([program, arguments])


# Matrix to Array
A = np.asarray(img).reshape(-1)

img.shape[0]
img.shape[1]
img.shape[2]

'''