#!/usr/bin/python

import sys, getopt
from Utils import *
from Histogram import *
from Thresholding import *
from Quantization import *


def main(argv):
    inimage = ''
    outimage = ''
    algorithm = ''
    params = []
    channels = 0

    try:
        opts, args = getopt.getopt(argv, "hi:c:o:a:p:", ["inimage=", "channels=", "outimage=", "alg", "params"])
    except getopt.GetoptError:
        print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -a <algorithm> -p [param1,param2,...]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: test.py -i <inimage> -c <numberchannels> -o <outimage> -a <algorithm> -p [param1,param2,...]')
            sys.exit()
        elif opt in ("-i", "--inimage"):
            inimage = arg
        elif opt in ("-c", "--channels"):
            channels = int(arg)
        elif opt in ("-o", "--outimage"):
            outimage = arg
        elif opt in ("-a", "--alg"):
            algorithm = arg
        elif opt in ("-p", "--params"):
            params = list(map(float, arg.strip('[]').split(',')))

    img_pil = open_image(inimage, channels)
    img_np = pil_to_np(img_pil)
    img_out = None
    save = False

    if algorithm == "gamma_correction":
        img_out = correction_gamma(img_np, params[0], channels)
        save = True
    elif algorithm == "global_limiarization":
        img_out = global_limiarization(img_np, np.array(params))
        save = True
    elif algorithm == "otsu_thresholding":
        img_out, _, _ = otsu_thresholding(img_np)
        save = True
    elif algorithm == "equalize_histogram":
        img_out = equalize_histogram(img_np)
        save = True
    elif algorithm == "popularity_algorithm":
        img_out = popularity_algorithm(img_pil, int(params[0]), int(params[1]), channels)
        save = True
    elif algorithm == "show_histogram":
        compute_histogram(img_np, int(params[1]),  channels, True)
    elif algorithm == "show_image":
        show_image(img_np, channels)

    if save:
        img_out = np_to_pil(img_out)
        save_image(img_out, outimage)


if __name__ == "__main__":
    main(sys.argv[1:])


