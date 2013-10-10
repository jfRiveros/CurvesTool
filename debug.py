import math
import random
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(file_path):
	img_array = scipy.misc.imread(file_path)
	return img_array

def rgb2gray(img_rgb):
	R_CONST = 0.299
	G_CONST = 0.587
	B_CONST = 0.114
	img_shape = img_rgb.shape
	M = img_shape[0]
	N = img_shape[1]
	K = img_shape[2]
	#declare empty array for grayscale values
	# img_gray = np.zeros([M,N])
	# for m in range(M):
	# 	for n in range(N):
	# 		img_rgb_r = img_rgb[m,n,0]
	# 		img_rgb_b = img_rgb[m,n,1]
	# 		img_rgb_g = img_rgb[m,n,2]
	# 		img_gray[m,n] = (R_CONST * img_rgb_r) + (G_CONST * img_rgb_g) + (B_CONST * img_rgb_b)
	# return img_gray
	vg_filter = np.array([0.299, 0.587, 0.114])
	img_gray = np.dot(img_rgb[...,:3], vg_filter)
	max_val = np.max(img_gray)
	img_gray = (img_gray/max_val) * 255.0
	return img_gray

def plot_image(img_array):
	plt.imshow(img_array, cmap = 'gray')

def hist_256(img_array):
	#initialize histogram vector
	hist_vector = np.zeros(256)
	for i in range(img_array.shape[0]):
		for j in range(img_array.shape[1]):
			pixel_val = int(round(img_array[i,j]))
			hist_vector[pixel_val] = hist_vector[int(pixel_val)] + 1
	return hist_vector

def plot_hist(hist):
	plt.hist(hist, 256)

if __name__=="__main__":
	a = load_image('87.png')
	a = rgb2gray(a)
	b = hist_256(a)
	print b
	plot_hist(b)
	# plt.subplot(221), plot_image(a)
	# plt.subplot(222), plot_hist(b), plt.ylim((0,1000)), plt.xlim((0,255))
	plt.show()
