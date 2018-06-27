from Compression import *
from Utils import *

channels = 3

img = pil_to_np(open_image("cat.jpg", channels));
print(img.shape)
huffcoding = HuffmanCode()
code = huffcoding.encode(img)
huffcoding.save("cu", code)


