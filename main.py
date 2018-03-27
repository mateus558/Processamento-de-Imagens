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

def imageMenu():
    clear()

    while True:
        print("1 - Open Image")
        print("2 - Save Image")
        print("3 - Show Image")

        o = int(input(">"))
        clear()
        if o == 1:
            path = input("Image path: ")
            img = np.asarray(Image.open(path), dtype=np.uint8)
        elif o == 2:
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil.save("out.jpg")
        elif o == 3:
            implt = plt.imshow(img)
            plt.show()
def main():
    clear()

    while True:
        print("1 - Image")

        o = int(input(">"))

        if o == 1:
            imageMenu()

main()