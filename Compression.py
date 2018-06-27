import heapq
import numpy as np
from PIL import Image
import sys
import struct
import bitarray

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
        self.shape = (0,0)

    def createHash(self, key):
        x = key[0] << 7
        for chr in key[1:]:
            x = ((1000003 * x) ^ chr) & (1<<32)
        return x
    
    def createHuffTree(self, image = None):
        if image is None:
            print("Need to provide a image to encode!")
            return ;

        pil_img = Image.fromarray(image)
        colors = pil_img.getcolors(pil_img.size[0]*pil_img.size[1])
        
        for color in colors:
            heapq.heappush(self.Q, HuffNode(pixel = color[1], freq = color[0]))
        while len(self.Q) > 1 :
            x = heapq.heappop(self.Q)
            y = heapq.heappop(self.Q)
            z = HuffNode(left = x, right = y, freq = (x.getFrequency() + y.getFrequency()))
            heapq.heappush(self.Q, z) 
        self.root = heapq.heappop(self.Q)
        
    def createCodingAux(self, root, code):
        if root is None:
            return
        if root.pixel is not None:
            self.encodedText += code
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
            
            byte = file.read(1)
            while(byte != ""):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)
            padded_info = bit_string[:8]
            extra_padding = int(padded_info, 2)
            padded_encoded_text = bit_string[8:]
            encoded_text = bit_string[:-1*extra_padding]

        return bit_string

    def encode(self, image = None):
        if image is None:
            return;

        self.encodedText = str("{0:b}".format(image.shape[0]))
        for i in range(len(self.encodedText), 16):
            self.encodedText += "0"
        self.encodedText += str("{0:b}".format(image.shape[1]))
        for i in range(len(self.encodedText), 32):
            self.encodedText += "0"

        self.createHuffTree(image)
        self.createCoding()
        
        b = bytearray()
        for i in range(0, len(self.encodedText), 8):
            byte = self.encodedText[i:i+8]
            b.append(int(byte, 2))
		
        return b        
           
    def decode(self, code):
        img = []
        cod = ''
        itr = self.root
       
        for bit in code:
            cod += bit
            if itr is not None and itr.pixel is not None:     
                img.append(np.uint8(np.array(itr.pixel)))
                itr = self.root
                cod = ''
            if bit == '1':
                if itr is not None and itr.right is not None:
                    itr = itr.right
            elif bit == '0':
                if itr is not None and itr.left is not None:
                    itr = itr.left
                    
        return np.array(img, dtype=np.uint8)