#!/usr/bin/python

import sys, getopt
#from Utils import *
from ErrorMetrics import *
from ColorSpaceTransformation import *
from Quantization import *


def main(argv):
    inimage = ''
    outimage = ''
    show = False
    algorithm = ''
    params = []
    channels = 0
    errors = False

    try:
        opts, args = getopt.getopt(argv, "hi:c:o:s:a:p:e:", ["inimage=", "channels=", "outimage=", "show=", "alg", "params", "errors"])
    except getopt.GetoptError:
        print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -s <show> -a <algorithm> -p [param1,param2,...] -e <errors>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -s <show> -a <algorithm> -p [param1,param2,...] -e <errors>')
            sys.exit()
        elif opt in ("-i", "--inimage"):
            inimage = arg
        elif opt in ("-c", "--channels"):
            channels = int(arg)
        elif opt in ("-o", "--outimage"):
            outimage = arg
        elif opt in ("-s", "--show"):
            show = True
        elif opt in ("-a", "--alg"):
            algorithm = arg
        elif opt in ("-p", "--params"):
            params = list(map(float, arg.strip('[]').split(',')))
        elif opt in ("-e", "--errors"):
            errors = True

    img_pil = open_image(inimage, channels)
    img_np = pil_to_np(img_pil)
    img_out = img_np
    save = False

    if algorithm == "gamma_correction":
        img_out = gamma_correction(img_out, params[0], channels)
        save = True
    elif algorithm == "global_limiarization":
        img_out = global_limiarization(img_out, np.array(params))
        save = True
    elif algorithm == "otsu_thresholding":
        img_out, _, _ = otsu_thresholding(img_out)
        save = True
    elif algorithm == "equalize_histogram":
        img_out = equalize_histogram(img_out)
        save = True
    elif algorithm == "popularity_algorithm":
        img_out = popularity_algorithm(img_pil, int(params[0]), int(params[1]), channels)
        show_image_np(img_out, channels)
        save = True
    elif algorithm == "mediancut_algorithm":
        img_out = mediancut_algorithm(img_pil, int(params[0]), channels)
        save = True
    elif algorithm == "quantization":
        img_out = quantization(img_np, channels, int(params[0]), int(params[1]), int(params[2]))
        if np.all(np.array((int(params[0]), int(params[1]), int(params[2]))) <= 8):
            save = True
    elif algorithm == "show_histogram":
        compute_histogram(img_np, int(params[0]),  channels, True)
    elif algorithm == "show_image":
        show_image_np(img_np, channels)

    if errors:
        img_np = pil_to_np(img_pil)
        signal_to_noise_ration(img_np, img_out, channels)

    if save:
        img_out = np_to_pil(img_out)
        save_image(img_out, outimage)
        if show:
            show_image_np(img_out, channels)


if __name__ == "__main__":
    main(sys.argv[1:])


