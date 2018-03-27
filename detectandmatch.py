import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
from scipy.ndimage import filters
import scipy.ndimage as nd
from numpy import linalg as LA
import sys
from scipy import signal
import math
from numpy.lib.stride_tricks import as_strided
import scipy.ndimage as ndimage

def detect_features(image, window_size):
"""
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
"""
	cornerList = []
	img_gry = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	#normalize image with gaussian blur
	blurred_img = cv2.medianBlur(img_gry, 11)
	sigma = 1
	
	#This is to reduce the noise in the image
	Gaus_Ker = makeGaussian(15, sigma, center=None)
        
	# makes the img_gry blur by convolving this gaussian kernel.
	img_gry = signal.convolve2d(img_gry, Gaus_Ker, boundary='symm', mode='same', fillvalue=0)

    	#Using sobel filter to get better derivative values
	# s_mask = 17
	# sobelx = np.abs(cv2.Sobel(img_gry, cv2.CV_64F, 1, 0, ksize=s_mask))
	#b_sobelx = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=s_mask))
	# sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
	#b_sobelx = interval_mapping(b_sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
	# sobely = np.abs(cv2.Sobel(img_gry,cv2.CV_64F,0,1,ksize=s_mask))
	# sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
	#b_sobely = np.abs(cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=s_mask))
	#b_sobely = interval_mapping(b_sobely, np.min(sobely), np.max(sobely), 0, 255)
	# sobel_xy = 0.5 * sobelx + 0.5 * sobely
	#b_sobel_xy = 0.5 * b_sobelx + 0.5 * b_sobely
	
	# gradient finding
	d_x, d_y = np.gradient(img_gry, edge_order=2)
	# folllowed some codes from https://github.com/hughesj919/HarrisCorner/blob/master/Corners.py
    	
	# making the H matrix
	Ixx = d_x**2
	Ixy = 2*(d_x * d_y)
	Iyy = d_y**2
	k = 0.05

	rows = image.shape[0]
	cols = image.shape[1]
	newImg = img_gry.copy()
	newImg=newImg.astype(np.float32)
	color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
	offset = window_size/2
	Spring_Green = (0,255,127)
	print "Finding Corners..."

	for y in range(offset, cols+1):
		for x in range(offset, rows-offset):
            	#Calculate sum of squares
			windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
			Sxx = windowIxx.sum()
			Sxy = windowIxy.sum()
			Syy = windowIyy.sum()

           	 	#Find determinant and trace, use to get corner response
			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - k*(trace**2)
			#print r.max()

            		#If corner response is over threshold, color the point and add to corner list
			if r > 55000000:
				# print x, y, r
				cornerList.append([x, y, r])
				# color_img.itemset((y, x, 0), 0)
				# color_img.itemset((y, x, 1), 0)
				# color_img.itemset((y, x, 2), 255)
				cv2.circle(color_img, (int(x), int(y)), 2, Spring_Green)
	return color_img, cornerList

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def makeGaussian(size, sigma, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

# with some help from my colleague HyungMu Lee the following code matches detected features from one image1 to features in image 2
def match_features(feature_coords1, feature_coords2, image1, image2):
	I1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	heig1, wid1 = np.shape(I1)
	heig2, wid2 = np.shape(I2)

    	# Patch Size
	window_size = 17
	
	print "matching features..."
	matches_1 = {}
	for i in range(len(feature_coords1)):
		row1 = feature_coords1[i][0]
		col1 = feature_coords1[i][1]

		max_val1 = 0
		max_ncc1 = -1

		#define patch's shape
		window_1 =np.zeros([window_size, window_size], float)
		size = (window_size - 1)/2

		# Avoid first and last axis so that there are no wrong matches
		if (row1 - size > 0 and col1 - size >0) and (row1 + size < heig1 and col1 + size < wid1):
			window_1 = I1[(row1 - size):(row1 + size + 1), (col1 - size):(col1+size +1)]
			window_1 = window_1-window_1.mean()
			window_1 = window_1/window_1.std()

			for j in range(len(feature_coords2)):
				row2 = feature_coords2[j][0]
				col2 = feature_coords2[j][1]
				window_2 = np.zeros([window_size, window_size], float)
				if (row2 - size > 0 and col2 - size > 0) and (row2 + size < heig2 and col2 + size < wid2):
					window_2 = I2[(row2 - size):(row2 + size + 1), (col2 - size):(col2 + size + 1)]
					window_2 = window_2 - window_2.mean()
					window_2 = window_2/window_2.std()

					NCC1 = 0
					NCC1 = (window_1*window_2).sum()/window_size**2
					if NCC1 > max_ncc1:
						max_ncc1 = NCC1
						max_val1 = j
		matches_1[i] = max_val1

	matches_2 = {}
	for i in range(len(feature_coords2)):
		row2 = feature_coords2[i][0]
		col2 = feature_coords2[i][1]
		max_val2 = 0
		max_ncc2 = -1
		window_2 = np.zeros([window_size, window_size], float)
		size = (window_size - 1)/2
		if (row2 - size > 0 and col2 - size >0) and (row2 +size < heig2 and col2 + size < wid2):
			window_2 = I2[(row2 - size):(row2 + size + 1), (col2 - size):(col2 + size +1)]
			window_2 = window_2 - window_2.mean()
			window_2 = window_2/ window_2.std()

			for j in range(len(feature_coords1)):
				row1 = feature_coords1[j][0]
				col1 = feature_coords1[j][1]
				window_1 = np.zeros([window_size, window_size], float)
				if (row1 - size > 0 and col1 - size >0) and (row1+size < heig1 and col1 + size < wid1):
					window_1 = I1[(row1 - size):(row1 + size +1), (col1 - size):(col1 + size +1)]
					window_1 = window_1 - window_1.mean()
					window_1 = window_1/window_1.std()

					NCC2 = 0
					NCC2 = (window_1*window_2).sum() / window_size**2
					if NCC2 > max_ncc2:
						max_ncc2 = NCC2
						max_val2 = j
		matches_2[i] = max_val2

	match_img = np.concatenate((image1, image2), axis = 1)

	for val2 in matches_1:
		c1 = feature_coords1[val2][1]
		r1 = feature_coords1[val2][0]
		cv2.circle(match_img, (c1, r1), 2, (180, 255, 20), 3)

	for val1 in matches_2:
		c2 = feature_coords2[val1][1]
		r2 = feature_coords2[val1][0]
		cv2.circle(match_img, (c2 + image1.shape[1], r2), 2, (180, 255, 20), 3)

	matches = list()
	for i,j in matches_1.iteritems():
		m1 = matches_2[j]
		if i == m1:
			r1 = feature_coords1[i][0]
			c1 = feature_coords1[i][1]
			r2 = feature_coords2[j][0]
			c2 = feature_coords2[j][1] + image1.shape[1]
			matches.append([(r1,c1), (r2,c2)])
			cv2.line(match_img, (c1, r1), (c2, r2), (255, 255, 255), thickness=1, lineType=8, shift=0)
	return matches, match_img

def main():
	image =[]
	features_set = []
	neighbors_set = []
	img2_2 = Image.open('wall3.png')
	img2_2 = img2_2.resize((1000, 700), PIL.Image.ANTIALIAS)
	img2_2.save('wall3_resized.png')

	img1 = cv2.imread('wall1.png')
	img2 = cv2.imread('wall3_resized.png')	
	
	image.append(img1)
	image.append(img2)

	width = image[0].shape[1]
	height = image[0].shape[0]

	# harris_corner = np.zeros((height , 2*width  ,3) ,np.uint8)
	# harris_corner_desired = np.zeros((height , 2*width  ,3) ,np.uint8)
	# full_image = np.zeros((height , 2*width, 3) ,np.uint8)

	finalImg, cornerList = detect_features(img1, 2) # outputs more corners
	cv2.imwrite("finalimage.png", finalImg)
    
	finalImg2, cornerList2 = detect_features(img2, 2) # outputs less corners
	cv2.imwrite("finalimage2.png", finalImg2)

	matches, match_img = match_features(cornerList, cornerList2, img1, img2)
	cv2.imwrite('matched_img.png', match_img)

if __name__ == "__main__":
    main()
