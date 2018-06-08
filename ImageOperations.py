import subprocess
import numpy as np
from numpy import fft
from Utils import *
import scipy.special, scipy.signal
import matplotlib.pyplot as plt
from ErrorMetrics import *
from ctypes import *

dll = CDLL('Fourier-Transform/bin/Debug/libFourier-Transform.dll')

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

	degree *= 2	#Because is a separete filter
	return box_filter

def gaussian_filter_generator(dimension):
	gaussian_filter = [dimension]
	for i in range(dimension):
		gaussian_filter[i] = special.comb(dimension, i)	#or special.binom(dimension, i)

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
		return filter, soma

	elif selected == 1:
		filter = [[-1, -2, -1],
				  [ 0,  0,  0],
				  [ 1,  2,  1]]
		return filter

	elif selected == 2:	# It needs to be completed
		filter = [[ 1,  0],
				  [ 0, -1]]
		return filter

	elif selected == 3:
		filter = [[0,  1,  0],
				  [1, -4,  1],
				  [0,  1,  0]]
		return filter

#
#	filter:
#		- 1: ideal_high_pass_filter
#		- 2: ideal_low_pass_filter
#
def ideal_pass_filter(img_real, img_imag, width, height, radius, filter):
	center_x = int(height / 2)
	center_y = int(width / 2)
	for i in range(0, height):
		for j in range (0, width):
			distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)

			if (filter == 1 and distance > radius) or (filter == 2 and distance <= radius):
				if i < height and i >= 0 and j < width and j >= 0:
					x = (i*width*3)+(j*3)
					img_real[x]   = 0
					img_real[x+1] = 0
					img_real[x+2] = 0

					img_imag[x]   = 0
					img_imag[x+1] = 0
					img_imag[x+2] = 0

	return img_real, img_imag


def fourier_transform_scipy(img_np):
	img_as_array = np.asarray(img_np).reshape(-1)
	img_as_array_out = np.fft.fft(img_as_array)

	img_inverse_out = np.fft.ifft(img_as_array_out)

	img_np_inverse_out = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)

	width = img_np.shape[1]

	for u in range (img_np.shape[0]):
		for v in range (img_np.shape[1]):
			i = (u*width*3)+(v*3)
			img_np_inverse_out[u][v][0] = float(img_inverse_out[i])
			img_np_inverse_out[u][v][1] = float(img_inverse_out[i+1])
			img_np_inverse_out[u][v][2] = float(img_inverse_out[i+2])


	print('\nFT Scipy')

	mean_square_error(img_np, img_np_inverse_out, img_np.shape[2])

	signal_to_noise_ration(img_np, img_np_inverse_out, img_np.shape[2])


#
#	algorithm:
#		- 0: do nothing
#		- 1: ideal high pass filter
#		- 2: ideal low pass filter
#
def fourier_transform(img_np, filter=0, radius=5):
	img_aux = np.asarray(img_np).reshape(-1)

	img_as_list = list(img_aux)
	img_as_list = (c_int * len(img_as_list)) (*img_as_list)

	img_real_ft = [0] * len(img_as_list)
	img_real_ft = (c_double * len(img_as_list)) (*img_as_list)

	img_imag_ft = [0] * len(img_as_list)
	img_imag_ft = (c_double * len(img_as_list)) (*img_as_list)

	print('\nFT Implemented')


	dll.Fourier_transform(img_as_list, c_int(img_np.shape[0]), c_int(img_np.shape[1]), img_real_ft, img_imag_ft)

	magnitude = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
	phase_angle = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)

	width = img_np.shape[1]

	for u in range (img_np.shape[0]):
		for v in range (img_np.shape[1]):
			i = (u*width*3)+(v*3)
			magnitude[u][v][0] = np.uint8(np.sqrt(img_real_ft[i]**2 + img_imag_ft[i]**2))
			magnitude[u][v][1] = np.uint8(np.sqrt(img_real_ft[i+1]**2 + img_imag_ft[i+1]**2))
			magnitude[u][v][2] = np.uint8(np.sqrt(img_real_ft[i+2]**2 + img_imag_ft[i+2]**2))
			if img_real_ft[i] == 0.0:
				if img_imag_ft[i] >= 0.0:
					phase_angle[u][v][0] = 90.0
				else:
					phase_angle[u][v][0] = -90.0

				if img_imag_ft[i+1] >= 0.0:
					phase_angle[u][v][1] = 90.0
				else:
					phase_angle[u][v][1] = -90.0

				if img_imag_ft[i+2] >= 0.0:
					phase_angle[u][v][2] = 90.0
				else:
					phase_angle[u][v][2] = -90.0
			else:
				phase_angle[u][v][0] = np.uint8(np.arctan(img_imag_ft[i] / img_real_ft[i]))
				phase_angle[u][v][1] = np.uint8(np.arctan(img_imag_ft[i+1] / img_real_ft[i+1]))
				phase_angle[u][v][2] = np.uint8(np.arctan(img_imag_ft[i+2] / img_real_ft[i+2]))

	save_image(magnitude, 'magnitude.jpg')
	save_image(phase_angle, 'phase_angle.jpg')

	if filter != 0:
		img_real_ft, img_imag_ft = ideal_pass_filter(img_real_ft, img_imag_ft, img_np.shape[0], img_np.shape[1], radius, filter)

	for u in range (img_np.shape[0]):
		for v in range (img_np.shape[1]):
			i = (u*width*3)+(v*3)
			magnitude[u][v][0] = np.uint8(np.sqrt(img_real_ft[i]**2 + img_imag_ft[i]**2))
			magnitude[u][v][1] = np.uint8(np.sqrt(img_real_ft[i+1]**2 + img_imag_ft[i+1]**2))
			magnitude[u][v][2] = np.uint8(np.sqrt(img_real_ft[i+2]**2 + img_imag_ft[i+2]**2))
			if img_real_ft[i] == 0.0:
				if img_imag_ft[i] >= 0.0:
					phase_angle[u][v][0] = 90.0
				else:
					phase_angle[u][v][0] = -90.0

				if img_imag_ft[i+1] >= 0.0:
					phase_angle[u][v][1] = 90.0
				else:
					phase_angle[u][v][1] = -90.0

				if img_imag_ft[i+2] >= 0.0:
					phase_angle[u][v][2] = 90.0
				else:
					phase_angle[u][v][2] = -90.0
			else:
				phase_angle[u][v][0] = np.uint8(np.arctan(img_imag_ft[i] / img_real_ft[i]))
				phase_angle[u][v][1] = np.uint8(np.arctan(img_imag_ft[i+1] / img_real_ft[i+1]))
				phase_angle[u][v][2] = np.uint8(np.arctan(img_imag_ft[i+2] / img_real_ft[i+2]))

	save_image(magnitude, 'magnitude2.jpg')
	save_image(phase_angle, 'phase_angle2.jpg')
	
	#Inverse

	img_inverse_out = [0] * len(img_aux)
	img_inverse_out = (c_int * len(img_inverse_out)) (*img_inverse_out)

	dll.Inverse_fourier_transform(img_real_ft, img_imag_ft, img_inverse_out)

	img_np_inverse_out = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)

	for u in range (img_np.shape[0]):
		for v in range (img_np.shape[1]):
			i = (u*width*3)+(v*3)
			img_np_inverse_out[u][v][0] = img_inverse_out[i]
			img_np_inverse_out[u][v][1] = img_inverse_out[i+1]
			img_np_inverse_out[u][v][2] = img_inverse_out[i+2]


	save_image(img_np_inverse_out, 'inverse_out.jpg')

	mean_square_error(img_np, img_np_inverse_out, img_np.shape[2])

	signal_to_noise_ration(img_np, img_np_inverse_out, img_np.shape[2])

