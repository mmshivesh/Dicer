import math
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pprint import pprint


parser = ap.ArgumentParser(description='Convert a picture into a image made with Dice!')
parser.add_argument('Filename', help='Name of the image file to convert')
parser.add_argument('-r',dest='Resolution', help='Factor to pixelate the image/Size of a die')
parser.add_argument('-V',dest='Verbose', help='Run in diagnostic mode/verbose mode', action='store_true')

args = parser.parse_args()


## Utility code here ##
jumpsize = int(args.Resolution)
image = imread(args.Filename)
if args.Verbose:
	print("Original Dimensions are : ", image.shape)

def printDiced(array, r,c):
	for i in range(r):
		for j in range(c):
			print(array[i][j], end=' ')
		print()

def averageGray(subset):		# Take a subset matrix and assign it with average intensity value of the subset and return it
	sum = 0
	with np.nditer(subset) as it:
		for x in it:
			sum = sum + x
	with np.nditer(subset, op_flags = ['readwrite']) as it:
		for x in it:	
			x[...] = sum/subset.size
	return sum/subset.size

def convert2gray(image):		# Convert the passed image to grayscale and return the image
	if image.shape[2] == 3:
		grayscaleFactor = [0.299, 0.587, 0.114]
	elif image.shape[2] == 4:
		grayscaleFactor = [0.299, 0.587, 0.114, 0]
	grayImg = image.dot(grayscaleFactor)
	return grayImg

grayscaleImage = np.array(convert2gray(image))
rows,cols = grayscaleImage.shape
if args.Verbose:
	print("Grayscale Dimensions are : ", grayscaleImage.shape)

pixelatedGrayIm = [[] for i in range(math.ceil(rows/jumpsize))]
ptr = 0
for i in range(0,rows,jumpsize):
	for j in range(0,cols,jumpsize):
		pixelatedGrayIm[ptr].append(averageGray(grayscaleImage[i:i+jumpsize, j:j+jumpsize]))
	ptr+=1
pixelatedGrayImNP = np.array(pixelatedGrayIm)
minrows, mincols = pixelatedGrayImNP.shape
if args.Verbose:
	print("Pixelated Dimensions are : ", pixelatedGrayImNP.shape)


maxval = np.amax(pixelatedGrayImNP)
minval = np.amin(pixelatedGrayImNP)
bucketsize = (maxval-minval)/6
dicedImage = [[] for i in range(minrows)]
dicedIndex = 0
for i in range(minrows):
	for j in range(mincols):
		if pixelatedGrayImNP[i, j] >= minval and pixelatedGrayImNP[i, j] < (minval + bucketsize) :
			dicedImage[dicedIndex].append(6)
		if pixelatedGrayImNP[i, j] >= (minval + bucketsize) and pixelatedGrayImNP[i, j] < (minval + 2*bucketsize) :
			dicedImage[dicedIndex].append(5)
		if pixelatedGrayImNP[i, j] >= (minval + 2*bucketsize) and pixelatedGrayImNP[i, j] < (minval + 3*bucketsize) :
			dicedImage[dicedIndex].append(4)
		if pixelatedGrayImNP[i, j] >= (minval + 3*bucketsize) and pixelatedGrayImNP[i, j] < (minval + 4*bucketsize) :
			dicedImage[dicedIndex].append(3)
		if pixelatedGrayImNP[i, j] >= (minval + 4*bucketsize) and pixelatedGrayImNP[i, j] < (minval + 5*bucketsize) :
			dicedImage[dicedIndex].append(2)
		if pixelatedGrayImNP[i, j] >= (minval + 5*bucketsize) and pixelatedGrayImNP[i, j] <= maxval :
			dicedImage[dicedIndex].append(1)
	dicedIndex+=1

printDiced(dicedImage, minrows, mincols)
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.subplot(1,2,2)
# plt.imshow(grayscaleImage, cmap = plt.get_cmap('gray'))
# plt.show()