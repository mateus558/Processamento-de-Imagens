from Compression import *
from Utils import *

channels = 3
img_name = "cat"
format = ".jpg"

img = pil_to_np(open_image(img_name + format, channels));
img_out = np.uint8(cosine_transform(img))

encoder = HuffmanCode()
code = encoder.encode(img_out)

encoder.save(img_name, code)
binary = encoder.open(img_name)

dimg = encoder.decode(binary)
#dimg = inverse_cosine_transform(img_out)

show_image_np(dimg, channels)