import os
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def clear():
    if platform == "linux" or platform == "linux2":
        os.system('clear')
    elif platform == "darwin":
        os.system('clear')
    elif platform == "win32":
        os.system('cls')
    return

"""def put_alpha_channel(img):
    img.putalpha(1)"""


def is_grey_scale(image):
    img = image.convert('RGB')
    width, height = img.size
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i,j))
            if r != g != b: return False
    return True


def conversor_image(img):
    if(is_grey_scale(img)):
        return img.convert('RGB')
    else:
        return img.convert('L')


def save_image(img):
    if(img.format == 'JPG'):
        img.save('out.jpg')
    else:
        img.save('out.png')


def squared_error(img1_np, img2_np):
    width = np.size(img1_np,1)
    height = np.size(img1_np,2)

    mean = 0
    for i in range(width):
        for j in range(height):
            for k in range(0,3):
                mean += (img1_np[i][j][k] - img2_np[i][j][k]) ** 2

    return mean


def mean_square_error(img1, img2):
    width, height = img1.size
    width2, height2 = img2.size

    if width == width2 and height == height2:
        img1_np = np.asarray(img1, dtype = np.uint8)
        img2_np = np.asarray(img2, dtype = np.uint8)

        return squared_error(img1_np, img2_np) / (width * height)
    else:
        return 0


def signal_to_noise_ration(original_img, noisy_img):
    width, height = original_img.size
    width2, height2 = noisy_img.size

    if width == width2 and height == height2:
        original_img_np = np.asarray(original_img, dtype = np.uint8)
        noisy_img_np = np.asarray(noisy_img, dtype = np.uint8)

        signal = 0
        for i in range(width):
            for j in range(height):
                for k in range(0,3):
                    signal += (noisy_img_np[i][j][k]) ** 2

        return signal / squared_error(original_img_np, noisy_img_np)
    else:
        return 0


def image_menu():
    clear()
    img = Image
    while True:
        print("1 - Open Image")
        print("2 - Save Image")
        print("3 - Show Image")

        o = int(input(">"))
        clear()
        if o == 1:
            # path = input("Image path: ")
            img = Image.open('C:/Users/kevyn/Desktop/Trabalho-1-de-processamento-de-Imagens/teste.jpg')
            #img.show()
            # img = np.asarray(img2, dtype=np.uint8)
        elif o == 2:
            #img_pil = Image.fromarray(img.astype(np.uint8))
            save_image(img)
        elif o == 3:
            # implt = plt.imshow(img)
            # plt.show()
            img.show()
        elif o == 4:
            img1 = Image.open('C:/Users/kevyn/Desktop/Trabalho-1-de-processamento-de-Imagens/teste1.jpg')
            img2 = Image.open('C:/Users/kevyn/Desktop/Trabalho-1-de-processamento-de-Imagens/teste2.jpg')
            print(mean_square_error(img1, img2))


def main():
    clear()

    while True:
        print("1 - Image")
        o = int(input(">"))
        if o == 1:
            image_menu()


main()
