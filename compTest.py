from Compression import *
from Utils import *

channels = 1

img = pil_to_np(open_image("cat.jpg", channels));
huffcoding = HuffmanCode()
code = huffcoding.encode(img)
#huffcoding.save("boi", code)
#huffcoding.open('oi')
#dimg = huffcoding.decode(code)
#show_image_np(dimg, channels)

