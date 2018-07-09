import heapq
import numpy as np
import collections

def entropy(frequency):
    sum = 0
    entropy = 0
    for key, freq in frequency.items():
        sum += freq
    for key, freq in frequency.items():
        px = freq / sum
        entropy += px * np.log2(px)
    return -1*entropy


def hash(string):
    x = ord(string[0]) << 7
    for chr in string[1:]:
        x = ((1000003 * x) ^ ord(chr)) & (1<<32)
    return x


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

    def __lt__(self, other):
        return self.frequency < other.frequency


class HuffmanCode:
    def __init__(self):
        self.Q = []
        self.codes = {}
        self.codes_ = np.zeros(256, dtype=np.int)
        self.mapping = {}
        self.root = None
        self.image = None
        self.encodedText = ''
        self.rows = 0
        self.cols = 0
        self.depth = 0
        self.infobits = 0
        self.frequency = None
    
    def createHuffTree(self, flatten_image = None):
        self.frequency = collections.Counter(flatten_image)

        for key, freq in self.frequency.items():
            heapq.heappush(self.Q, HuffNode(left = None, right = None, pixel = key, freq = freq))
                
        while len(self.Q) > 1 :
            x = heapq.heappop(self.Q)
            y = heapq.heappop(self.Q)
            z = HuffNode(left = x, right = y, freq = (x.getFrequency() + y.getFrequency()))
            heapq.heappush(self.Q, z) 
        self.root = heapq.heappop(self.Q)
        
    def createCodingAux(self, root, top):
        if root is None:
            return
        if not root.left and not root.right:
            code = ""
            for i in range(top):
                code = code + "{0}".format(self.codes_[i])
            self.codes[root.pixel] = code
            self.mapping[code] = root.pixel
            return;
    
        if root.left:
            self.codes_[top] = 0
            self.createCodingAux(root.getLeft(), top + 1)
        if root.right:
            self.codes_[top] = 1
            self.createCodingAux(root.getRight(), top + 1)

    def createCoding(self):
        top = 0
        self.createCodingAux(self.root, top)
    
    def encode_img_info(self, image = None):
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
        
        return sdepth + srows + scols

    def encode(self, image = None):
        if image is None or len(image) < 2:
            print("Need to provide a image to encode!")
            return ;
        
        flatten = np.asarray(image).reshape(-1)
        
        self.image = flatten
        self.createHuffTree(self.image)
        print("Huffman tree created.")
        self.createCoding()
        print("Huffman code created.")
        
        print("Encoding image.")

        
        for pixel in self.image:
            code = self.codes[pixel]
            self.encodedText += code

        img_info = self.encode_img_info(image)
        extra_padding = 8 - len(self.encodedText) % 8
        padded_info = "{0:08b}".format(extra_padding)

        for i in range(extra_padding):
            self.encodedText += "0"
   
        self.encodedText = padded_info + img_info + self.encodedText
        self.infobits += len(padded_info) + len(img_info)
        print("Image encoded.")

        size = self.rows*self.cols*self.depth
        print("\nImage Entropy: {0}".format(entropy(self.frequency)))
        print("Average number of bits by pixel: {0}".format(len(self.encodedText)/(size)))
        print("Compression rate: {0}%\n".format((1-len(self.encodedText)/(size*8))*100))
        
        print("Creating binary string.")
        b = bytearray()
        for i in range(0, len(self.encodedText), 8):
            byte = self.encodedText[i:i+8]
            b.append(int(byte, 2))
        print("Binary string created.")

        return b        

    def bytearray2string(self, _bytearray):
        string = ''
        for byte in _bytearray:
            string += '{0:08b}'.format(byte)
        return string

    def save(self, fname, code):
        newFile = open(fname+".msw", "wb")
        newFile.write(bytes(code))
        print("Binary written to file.")

    def open(self, fname):
        
        with open(fname+".msw", "rb") as binary_file:
            bit_string = ""
            
            byte = binary_file.read(1)
            while(len(byte) != 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = binary_file.read(1)
            padded_info = bit_string[:8]
            extra_padding = int(padded_info, 2)
            padded_encoded_text = bit_string[8:]
            encoded_text = bit_string[:-1*extra_padding][6:]
        
        print("Binary string read from file.")
        
        return encoded_text

    def decode(self, encoded_text):
        img_info = encoded_text[2:36]
        encoded_text = encoded_text[36:]

        self.depth = int(img_info[:2],2)
        self.rows = int(img_info[2:18],2)
        self.cols = int(img_info[18:36],2)

        img = []        
        if self.depth > 1:
            img = np.zeros((self.rows, self.cols, self.depth), dtype = np.uint8)
        else:
            img = np.zeros((self.rows, self.cols), dtype = np.uint8)
        
        pixels = []
        path = self.encodedText[self.infobits:]
        current_code = ""
        decoded_text = ""
        
        for bit in path:
            current_code += bit
            if(current_code in self.mapping):
                scalar = self.mapping[current_code]
                pixels.append(scalar)
                current_code = ""
        
        if self.depth == 1:		
            for i in range(self.rows):
                for j in range(self.cols):
                    img[i, j] = pixels[i*self.rows + j]
        else:
            for x in range(self.rows):
                for y in range(self.cols):
                    index = (x*self.cols*self.depth)+(y*self.depth)
                    img[x, y, 0] = pixels[index]
                    img[x, y, 1] = pixels[index+1]
                    img[x, y, 2] = pixels[index+2]
        
        return img 
