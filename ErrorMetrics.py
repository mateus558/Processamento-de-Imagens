import numpy as np
import math

def squared_error(img1_np, img2_np, channels):
    height = np.size(img1_np,0)
    width = np.size(img1_np,1)

    error = 0.0
    for i in range (height):
        for j in range (width):
            if channels == 1:
                error += ((img1_np[i][j] - img2_np[i][j]) ** 2)
            if channels == 3:
                for k in range (0,3):
                    img1_aux = np.int(img1_np[i][j][k])
                    img2_aux = np.int(img2_np[i][j][k])
                    error += ((img1_aux - img2_aux) ** 2)

    #print("Squared Error: " + str(error))
    return error


def mean_square_error(img1, img2, channels):
    height = np.size(img1, 0)
    width = np.size(img1, 1)

    height2 = np.size(img2, 0)
    width2 = np.size(img2, 1)

    if width == width2 and height == height2:
        mse = math.sqrt(squared_error(img1, img2, channels) / (width * height * channels))
        print ("MSE: " +  str(mse))
        return mse
    else:
        return 0


def signal_to_noise_ration(img, noisy_img, channels):
    height = np.size(img, 0)
    width = np.size(img, 1)

    height2 = np.size(noisy_img, 0)
    width2 = np.size(noisy_img, 1)

    if width == width2 and height == height2:
        #Version 1
        signal = 0.0
        for i in range(height):
            for j in range(width):
                if channels == 1:
                    signal += (noisy_img[i][j]) ** 2
                elif channels == 3:
                    for k in range (0,channels):
                        signal += (noisy_img[i][j][k]) ** 2
        print ("SNR: " + str(10 * np.log10 (signal / squared_error(img, noisy_img, channels))))
