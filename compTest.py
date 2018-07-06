from Compression import *
from Utils import *

channels = 3

img = pil_to_np(open_image("cat.jpg", channels));
img_out = cosine_transform(img)

huffcoding = HuffmanCode()
code = huffcoding.encode(img_out)

huffcoding.save("cat", code)
binary = huffcoding.open("cat")
dimg = huffcoding.decode(code)

dimg = inverse_cosine_transform(dimg)
dimg1 = inverse_cosine_transform(img_out)

show_image_np(dimg, channels)