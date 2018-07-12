import numpy as np
from ctypes import *
from Utils import *

block_size = 8

def cosine_transform(img, block_size=8, do_quantization=True, do_zig_zag_scan=True, do_DCPM=True):
    channels = img.shape[2]
    size = block_size * block_size * channels

    img_in_dct = [0] * size
    img_in_dct = (c_double * size) (*img_in_dct)

    img_out_dct = [0.0] * size
    img_out_dct = (c_double * size) (*img_out_dct)

    h = img.shape[0]
    w = img.shape[1]

    mod_height = h % block_size
    mod_width  = w % block_size

    img_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float)

    for i in range(0, h + mod_height, block_size):
        for j in range(0, w + mod_width, block_size):

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)

                    if(x+i < img.shape[0] and y+j < img.shape[1]):
                        img_in_dct[k]   = img[x+i][y+j][0]
                        img_in_dct[k+1] = img[x+i][y+j][1]
                        img_in_dct[k+2] = img[x+i][y+j][2]

                    else:
                        img_in_dct[k]   = 0.0
                        img_in_dct[k+1] = 0.0
                        img_in_dct[k+2] = 0.0

            dll.Cosine_transform(img_in_dct, img_out_dct)


            img_out_dct_aux = np.zeros((block_size, block_size, img.shape[2]), dtype=np.float)

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)
                    img_out_dct_aux[x][y][0] = img_out_dct[k]
                    img_out_dct_aux[x][y][1] = img_out_dct[k+1]
                    img_out_dct_aux[x][y][2] = img_out_dct[k+2]

            if(do_quantization):
                img_out_dct_aux = quantization(img_out_dct_aux, block_size, inverse=False)

            if(do_zig_zag_scan):
                img_out_dct_aux = zigzag(img_out_dct_aux)
                img_out_dct_aux = np.reshape(img_out_dct_aux, (block_size, block_size, channels))

            if(i == 0 and j == 0 and do_DCPM):
                out = DCPM(img_out_dct_aux, channels)

            for x in range(block_size):
                for y in range(block_size):
                    if(x+i < img.shape[0] and y+j < img.shape[1]):
                        img_out[x+i][y+j][0] = img_out_dct_aux[x][y][0]
                        img_out[x+i][y+j][1] = img_out_dct_aux[x][y][1]
                        img_out[x+i][y+j][2] = img_out_dct_aux[x][y][2]

    return img_out


def inverse_cosine_transform(img, do_inverse_quantization=True, do_inverse_zig_zag_scan=True, do_inverse_DCPM=True):
    channels = img.shape[2]
    size = block_size * block_size * channels

    img_in_dct = [0.0] * size
    img_in_dct = (c_double * size) (*img_in_dct)

    img_out_dct = [0.0] * size
    img_out_dct = (c_double * size) (*img_out_dct)

    h = img.shape[0]
    w = img.shape[1]

    mod_height = h % block_size
    mod_width  = w % block_size

    img_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float)

    block = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float)


    for i in range(0, h + mod_height, block_size):
        for j in range(0, w + mod_width, block_size):

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)

                    if(x+i < img.shape[0] and y+j < img.shape[1]):
                        block[x][y][0] = img[x+i][y+j][0]
                        block[x][y][1] = img[x+i][y+j][1]
                        block[x][y][2] = img[x+i][y+j][2]

                    else:
                        block[x][y][0] = 0.0
                        block[x][y][1] = 0.0
                        block[x][y][2] = 0.0

            if(do_inverse_zig_zag_scan):
                block_aux = inverse_zigzag(block.flatten(), block_size, block_size)
                block = np.reshape(block_aux, (block_size, block_size, channels)) 

            if(do_inverse_quantization):
                block = quantization(block, inverse=True)

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)

                    img_in_dct[k]   = block[x][y][0]
                    img_in_dct[k+1] = block[x][y][1]
                    img_in_dct[k+2] = block[x][y][2]


            dll.Inverse_cosine_transform(img_in_dct, img_out_dct)

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)
                    block[x][y][0] = img_out_dct[k]
                    block[x][y][1] = img_out_dct[k+1]
                    block[x][y][2] = img_out_dct[k+2]

            if(i == 0 and j == 0):
                print(block)

            for x in range(block_size):
                for y in range(block_size):
                    if(x+i < img.shape[0] and y+j < img.shape[1]):
                        img_out[x+i][y+j][0] = block[x][y][0]
                        img_out[x+i][y+j][1] = block[x][y][1]
                        img_out[x+i][y+j][2] = block[x][y][2]

    return img_out