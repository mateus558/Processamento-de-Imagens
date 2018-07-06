import heapq
import numpy as np
from PIL import Image
import sys
import collections
import struct
import bitarray
import os


def int2bin(s):
	b = str(s) if s<=1 else bin(s>>1) + str(s&1)
	return b.replace("0b", "")

class HuffNode:

	def __init__(self, pixel = None, left = None, right = None, freq = None):
		self.left = left
		self.right = right
		self.frequency = freq
		self.pixel = pixel
	
	def getLeft(self):
		return self.left

	def getRight(self):
		return self.right
	
	def getFrequency(self):
		return self.frequency
	
	def getPixel(self):
		return self.pixel

	def __comp__(self, other):
		return cmp(self.frequency, other.frequency)
	
	def __lt__(self, other):
		return self.frequency < other.frequency

class HuffmanCode:
    	def __init__(self):
		self.Q = []
		self.codes = {}
		self.mapping = {}
		self.root = None
		self.image = None
		self.encodedText = ''
		self.rows = 0
		self.cols = 0
		self.codes_array = []
		self.depth = 0
		self.infobits = 0
	
	def createHuffTree(self, flatten_image = None):
		frequency = collections.Counter(flatten_image)
		sum = 0
		for key, freq in frequency.items():
			heapq.heappush(self.Q, HuffNode(pixel = key, freq = freq))
			sum += freq
		print(sum)
		while len(self.Q) > 1 :
			x = heapq.heappop(self.Q)
			y = heapq.heappop(self.Q)
			z = HuffNode(left = x, right = y, freq = (x.getFrequency() + y.getFrequency()))
			heapq.heappush(self.Q, z) 
		self.root = heapq.heappop(self.Q)
		
	def createCodingAux(self, root, code):
		if root is None:
			return
		if not root.pixel is None:
			self.codes[root.pixel] = code
			self.mapping[code] = root.pixel
			return;
		code = code + "0"
		self.createCodingAux(root.getLeft(), code)
		code = code + "1"
		self.createCodingAux(root.getRight(), code)

	def createCoding(self):
		code = ""
		self.createCodingAux(self.root, code)

	def save(self, fname, code):
		newFile = open(fname+".bin", "wb")
		newFile.write(bytes(code))
	
	def open(self, fname):
		with open(fname+".bin", "rb") as binary_file:
			bit_string = ""
			
			byte = binary_file.read(1)
			while(byte != ""):
				byte = ord(byte)
				bits = bin(byte)[2:].rjust(8, '0')
				bit_string += bits
				byte = binary_file.read(1)
			padded_info = bit_string[:8]
			extra_padding = int(padded_info, 2)
			padded_encoded_text = bit_string[8:]
			encoded_text = bit_string[:-1*extra_padding]
		
		return encoded_text

	def encode(self, image = None):
		if image is None or len(image) < 2:
			print("Need to provide a image to encode!")
			return ;
		print(image.shape)
		self.cols = image.shape[1]
		self.rows = image.shape[0]
		
		sdepth = ''
		if len(image.shape) == 2:
			self.depth = 1
			sdepth = '01'   
		elif len(image.shape) == 3:
			self.depth = image.shape[2]
			sdepth = '11'
		
		srows = str("{0:0b}".format(self.rows))
		for i in range(len(srows), 16):
			srows = "0" + srows
		scols = str("{0:0b}".format(self.cols))
		for i in range(len(scols), 16):
			scols = "0" + scols
		info = sdepth + srows + scols
		self.infobits += len(info)
		self.image = image.flatten()
		self.createHuffTree(self.image)
		self.createCoding()
		print(self.codes)
		for pixel in self.image:
			self.codes_array.append(self.codes[pixel])

		for code in self.codes_array:
			self.encodedText += code

		self.encodedText = info + self.encodedText
		extra_padding = 8 - len(self.encodedText) % 8
		for i in range(extra_padding):
			self.encodedText += "0"
		self.infobits += extra_padding           

		padded_info = "{0:08b}".format(extra_padding)
		self.encodedText = padded_info + self.encodedText
		self.infobits += len(padded_info)
		b = bytearray()
		for i in range(0, len(self.encodedText), 8):
			byte = self.encodedText[i:i+8]
			b.append(int(byte, 2))

		return b        

	def bytearray2string(self, _bytearray):
		string = ''
		for byte in _bytearray:
			string += '{0:08b}'.format(byte)
		return string

	def decode(self, code):
		img = []
		cod = ''
		
		padded_info = code[0]
		extra_padding = padded_info
		padded_encoded_text = code[8:]
		encoded_text = self.bytearray2string(code[:-1*extra_padding])[6:]

		img_info = encoded_text[2:36]
		encoded_text = encoded_text[36:]
		print(encoded_text[:100])
		self.depth = int(img_info[:2],2)
		self.rows = int(img_info[2:18],2)
		self.cols = int(img_info[18:36],2)
		print([self.depth, self.rows, self.cols])
		if self.depth > 1:
			img = np.zeros((self.rows, self.cols, self.depth), dtype = np.uint8)
		else:
			img = np.zeros((self.rows, self.cols), dtype = np.uint8)
		
		pixels = []

		#path = self.encodedText[42:]
		print(self.encodedText[42:100])
		itr = self.root
		cod = ''
		k = 0
		path = ''
		for byte in self.codes_array:
			path += byte
		for i in range(0, len(path)):
			cod = cod + path[i]
			if cod in self.mapping:
				pixels.append(self.mapping[cod])
				cod = ''
		
		#for byte in self.codes_array:
		 #   pixels.append(self.mapping[byte])
		print(cod[:100])
				
		print(self.image[:100])
		print(pixels)

		for i in range(self.rows):
			for j in range(self.cols):
				img[i, j] = pixels[i*self.rows + j]
		return img
