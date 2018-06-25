#!/usr/bin/python

import sys, getopt
from ErrorMetrics import *
from ColorSpaceTransformation import *
from Quantization import *
from ImageOperations import *

def main(argv):
    inimage = ''
    outimage = ''
    show = False
    algorithm = ''
    params = []
    channels = 0
    errors = False

    try:
        opts, args = getopt.getopt(argv, 'hi:c:o:s:a:p:e:', ['inimage=', 'channels=', 'outimage=', 'show=', 'alg', 'params', 'errors'])
    except getopt.GetoptError:
        print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -s <show> -a <algorithm> -p [param1,param2,...] -e <errors>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -s <show> -a <algorithm> -p [param1,param2,...] -e <errors>')
            sys.exit()
        elif opt in ('-i', '--inimage'):
            inimage = arg
        elif opt in ('-c', '--channels'):
            channels = int(arg)
        elif opt in ('-o', '--outimage'):
            outimage = arg
        elif opt in ('-s', '--show'):
            show = int(arg)
        elif opt in ('-a', '--alg'):
            algorithm = arg
        elif opt in ('-p', '--params'):
            params = list(map(str, arg.strip('[]').split(',')))
        elif opt in ('-e', '--errors'):
            errors = True

    img_pil = open_image(inimage, channels)
    img_np = pil_to_np(img_pil)
    img_out = None
    save = False

    if algorithm == 'gamma_correction':
        img_out = gamma_correction(img_out, float(params[0]), channels)
        save = True

    elif algorithm == 'global_limiarization':
        img_out = global_limiarization(img_out, np.array(params))
        save = True

    elif algorithm == 'otsu_thresholding':
        if channels == 1:
            k, _ = otsu_thresholding(img_np)
            img_out = global_limiarization(img_np, k)
        else:
            kr, _ = otsu_thresholding(img_np[:][:][0])
            kg, _ = otsu_thresholding(img_np[:][:][1])
            kb, _ = otsu_thresholding(img_np[:][:][2])

            img_out = np.zeros(img_np.shape, dtype = np.uint8)

            img_out[:,:,0] = global_limiarization(img_np[:,:,0], kr)
            img_out[:,:,1] = global_limiarization(img_np[:,:,1], kg)
            img_out[:,:,2] = global_limiarization(img_np[:,:,2], kb)

            channels = 1
        save = True

    elif algorithm == 'equalize_histogram':
        img_out = equalize_histogram(img_out)
        save = True

    elif algorithm == 'popularity_algorithm':
        img_out = popularity_algorithm(img_pil, int(params[0]), int(params[1]), channels)
        show_image_np(img_out, channels)
        save = True

    elif algorithm == 'mediancut_algorithm':
        img_out = mediancut_algorithm(img_pil, int(params[0]), channels)
        save = True

    elif algorithm == 'quantization':
        img_out = quantization(img_np, channels, int(params[0]), int(params[1]), int(params[2]))
        if np.all(np.array((int(params[0]), int(params[1]), int(params[2]))) <= 8):
            save = True

    elif algorithm == 'show_histogram':
        compute_histogram(img_np, int(params[0]),  channels, True)

    elif algorithm == 'show_image':
        show_image_np(img_np, channels)
    elif algorithm == 'resize':
        img_out = resize(img_np, perc = float(params[0]), channels = channels, type = str(params[1]), kind = str(params[2]), img_name = outimage)
    elif algorithm == 'high_pass_filter':
        img_out = high_pass_filter(img_np, selected = int(params[0]), channels = channels)
        save = True
    elif algorithm == 'low_pass_filter':
        if str(params[0]) == 'box':
            filter = box_filter_generator(shape = (int(params[1]),int(params[1])))
        else:
            filter = np.array(gaussian_filter_generator(shape=(int(params[1]),int(params[1])),sigma=float(params[2])))
        img_out = convolve(img_np, filter, boundary='symm', channels = channels)
        save = True
    elif algorithm == 'blur_and_highlight':
        img_out = blur_and_highlight_filter(img_np, w = float(params[0]), dimension = int(params[1]))
        save = True
    elif algorithm == 'fourier_transform_scipy':
        fourier_transform_scipy(img_np, filter = float(params[0]), radius1 = float(params[1]), radius2 = float(params[2]), img_name = outimage)
        save = False
    elif algorithm == 'fourier_transform':
        fourier_transform(img_np, filter = int(params[0]), radius1 = int(params[1]), radius2 = int(params[2]), img_name = outimage)
        save = False
    elif algorithm == 'inverse_fourier_transform':
        img_real = open_image(str(params[0]), channels)
        img_imag = open_image(str(params[1]), channels)
        img_real = pil_to_np(img_real)
        img_imag = pil_to_np(img_imag)
        inverse_fourier_transform(img_np, imag_real, img_imag, outimage)
        save = False
    elif algorithm == 'convolve':
        kernel = open_image(str(params[0]), channels = int(params[1]))
        kernel = pil_to_np(kernel)
        img_out = convolve(img_np, kernel, channels = channels)
        save = True
    
    if errors:
        signal_to_noise_ration(img_np, img_out, channels)

    if show == 1:
        show_image_np(img_np, channels)

    if save:
        save_image(img_out, outimage)
        if show:
            show_image_np(img_out, channels)


if __name__ == '__main__':
    main(sys.argv[1:])


