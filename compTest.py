from Compression import *
from Utils import *
import time

channels = 3
img_name = "lena_gray"
format = ".bmp"

img = pil_to_np(open_image(img_name + format, channels));

start = time.time()
img_out = np.uint8(cosine_transform(img))

encoder = HuffmanCode()
code = encoder.encode(img_out)
end = time.time()

print("\n{0}s to encode image.\n".format(end - start))

encoder.save(img_name, code)
binary = encoder.open(img_name)

start = time.time()
dimg = encoder.decode(binary)
dimg = inverse_cosine_transform(img_out)
end = time.time()

print("\n{0}s to decode image.\n".format(end - start))


show_image_np(dimg, channels)