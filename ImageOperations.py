import subprocess
import numpy as np
import Utils
import scipy.special, scipy.signal

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

def low_pass_filter_generator(dimension, degree):
	box_filter = [dimension]
	for i in range(dimension):
		box_filter[i] = 1

	degree *= 2		#Because is a separete filter
	return box_filter

def gaussian_filter_generator(dimension):
	gaussian_filter = [dimension]
	for i in range(dimension):
		gaussian_filter[i] = special.comb(dimension, i) #or special.binom(dimension, i)

	return gaussian_filter

#
#	selected:
#		- 0: Prewitt
#		- 1: Sobel
#		- 2: Roberts 
#		- 3: Laplaciano 
#
def high_pass_filter(selected=0):
	if selected == 0:
		filter = [[-1, -1, -1],
				  [ 0,  0,  0],
				  [ 1,  1,  1]]
		return filter

	elif selected == 1:
		filter = [[-1, -2, -1],
				  [ 0,  0,  0],
				  [ 1,  2,  1]]
		return filter

	elif selected == 2:
		filter = [[ 1,  0],
				  [ 0, -1]]
		return filter

	elif selected == 3:
		# I think we have to calculate it (See: 'aula4-filtragem.pdf' page 50)
		#return filter
		return


def fourier_transform_scipy(img_np):
	img_as_array = np.asarray(img_np).reshape(-1)
	img_as_array_out = np.fft2(img_as_array)

	return img_as_array_out

#
#	algorithm:
#		- 0: fourier_transform_inverse
#		- 1: fourier_transform
#
def fourier_transform(img_np, algorithm=1):
	img_as_array = np.asarray(img_np).reshape(-1)

	program = Utils.get_path_name() + '/Fourier-Transform.exe'

	file_name = 'img_array.txt'
	np.savetxt(file_name, img_as_array, fmt='%d')

	arguments = (str(algorithm) + ' ' + str(img_np.shape[0]) + ' ' + str(img_np.shape[1]) + ' ' + str(img_np.shape[2]) + ' ' + file_name)
	print (arguments)
	subprocess.call([program, str(algorithm), str(img_np.shape[0]), str(img_np.shape[1]), str(img_np.shape[2]), file_name])
'''
	if subprocess.check_call(["ls", "-l"]):
		file_name_real = 'img_array_real.txt'
		file_name_imag = 'img_array_imag.txt'

		file = open(file_name_real, 'r').read()
		file = open(file_name_imag, 'r').read()

		img_real_np = [img_np.shape[0]][img_np.shape[1]][img_np.shape[2]]
		img_imag_np = [img_np.shape[0]][img_np.shape[1]][img_np.shape[2]]

		u = 0; v = 0; k = 0

		for value in file.split("\n"):
			img_real_np[u][v][k] = value

			k += 1

			if k == img_np.shape[2]:
				k = 0; v += 1
				if v == img_np.shape[1]:
					v = 0; u += 1

		u = 0; v = 0; k = 0

		for value in file.split("\n"):
			img_imag_np[u][v][k] = value

			k += 1

			if k == img_np.shape[2]:
				k = 0; v += 1
				if v == img_np.shape[1]:
					v = 0; u += 1


		magnitude = [img_np.shape[0]][img_np.shape[1]][img_np.shape[2]]
		phase_angle = [img_np.shape[0]][img_np.shape[1]][img_np.shape[2]]

		for i in range (img_np.shape[0]):
			for j in range (img_np.shape[1]):
				for k in range (img_np.shape[2]):
					magnitude[i][j][k] = np.sqrt(img_real_np[i][j][k]**2 + img_imag_np[i][j][k]**2)
					phase_angle[i][j][k] = np.arctan(img_imag_np[i][j][k] / img_real_np[i][j][k])
'''

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