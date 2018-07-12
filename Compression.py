from Utils import *
from ErrorMetrics import *
from ctypes import *
from Compression.cosine_transform import *
from Compression.quantization import *
from Compression.zig_zag_scan import *
from Compression.dcmp import *
from Compression.huffman import *

block_size = 8


def encode(img):
    dll = CDLL('Cosine-Transform/bin/Debug/libCosine-Transform.dll')
    img_coverted = rgb_to_ycbcr(img)

    channels = img_coverted.shape[2]
    size = block_size * block_size * channels

    img_in_dct = [0] * size
    img_in_dct = (c_double * size) (*img_in_dct)

    img_out_dct = [0.0] * size
    img_out_dct = (c_double * size) (*img_out_dct)

    h = img_coverted.shape[0]
    w = img_coverted.shape[1]

    mod_height = h % block_size
    mod_width  = w % block_size

    img_compress = np.zeros((h, w, channels), dtype=np.float)

    for i in range(0, h + mod_height, block_size):
        for j in range(0, w + mod_width, block_size):

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)

                    if(x+i < h and y+j < w):
                        img_in_dct[k]   = img_coverted[x+i][y+j][0]
                        img_in_dct[k+1] = img_coverted[x+i][y+j][1]
                        img_in_dct[k+2] = img_coverted[x+i][y+j][2]

                    else:
                        img_in_dct[k]   = 0.0
                        img_in_dct[k+1] = 0.0
                        img_in_dct[k+2] = 0.0


            dll.Cosine_transform(img_in_dct, img_out_dct)


            img_out_dct_aux = np.zeros((block_size, block_size, channels), dtype=np.float)

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)
                    img_out_dct_aux[x][y][0] = img_out_dct[k]
                    img_out_dct_aux[x][y][1] = img_out_dct[k+1]
                    img_out_dct_aux[x][y][2] = img_out_dct[k+2]


            img_out_dct_aux = quantization(img_out_dct_aux, block_size, inverse=False)


            img_out_dct_aux = zigzag(img_out_dct_aux)
            img_out_dct_aux = np.reshape(img_out_dct_aux, (block_size, block_size, channels))

            img_out_dct_aux[0][0][:] = DCPM(img_out_dct_aux[0][0][:])

            for x in range(block_size):
                for y in range(block_size):
                    if(x+i < h and y+j < w):
                        img_compress[x+i][y+j][0] = img_out_dct_aux[x][y][0]
                        img_compress[x+i][y+j][1] = img_out_dct_aux[x][y][1]
                        img_compress[x+i][y+j][2] = img_out_dct_aux[x][y][2]

    return img_compress

def decode(img):
    dll = CDLL('Cosine-Transform/bin/Debug/libCosine-Transform.dll')

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

    img_decoded = np.zeros((h, w, channels), dtype=np.float)

    block = np.zeros((block_size, block_size, channels), dtype=np.float)


    for i in range(0, h + mod_height, block_size):
        for j in range(0, w + mod_width, block_size):

            for x in range(block_size):
                for y in range(block_size):
                    k = (x*block_size*channels) + (y*channels)

                    if(x+i < h and y+j < w):
                        block[x][y][0] = img[x+i][y+j][0]
                        block[x][y][1] = img[x+i][y+j][1]
                        block[x][y][2] = img[x+i][y+j][2]

                    else:
                        block[x][y][0] = 0.0
                        block[x][y][1] = 0.0
                        block[x][y][2] = 0.0


            block[0][0][:] = inverse_DCPM(block[0][0][:])

            block_aux = inverse_zigzag(block.flatten(), block_size, block_size)
            block = np.reshape(block_aux, (block_size, block_size, channels)) 

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

            for x in range(block_size):
                for y in range(block_size):
                    if(x+i < h and y+j < w):
                        img_decoded[x+i][y+j][0] = block[x][y][0]
                        img_decoded[x+i][y+j][1] = block[x][y][1]
                        img_decoded[x+i][y+j][2] = block[x][y][2]

    img_converted = ycbcr_to_rgb(img_decoded)

    return img_converted


def compress(img, img_name):
    img_coded = encode(img)
    huffman = HuffmanCode()
    bin = huffman.encode(img_coded)
    rate = len(bin)/(img.shape[0]*img.shape[1]*img.shape[2]*8)
    print("Compress rate: {0} - {1}:1".format((1-rate)*100, 1/rate))
    huffman.save(img_name, bin)


def decompress(img_name):
    huffman = HuffmanCode()
    bin = huffman.open(img_name)
    img_coded = huffman.decode(bin)
    print(img_coded)
    img_decoded = decode(img_coded)
    signal_to_noise_ration(img, img_decoded, channels=3)
    save_image(img_decoded, img_name+' - Inverse_cosine_transform_out_2_huff.png')    
    
    return img_decoded

img_name = 'cat'
img_name_in = img_name+'.jpg'
huffman = HuffmanCode()


img = open_image(img_name_in, 3)
img = pil_to_np(img)
compress(img, img_name)
'''code = huffman.open(img_name)
img_coded = huffman.decode(code)
img_decoded = decode(img_coded)
signal_to_noise_ration(img, img_decoded, channels=3)
show_image_np(img_decoded, 3)
save_image(img_decoded, img_name+' - Inverse_cosine_transform_out_2_huff.png')    
'''
#show_image_np(img_decoded, 3)
print()
