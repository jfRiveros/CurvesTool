# import viz specific libraries
import viz
import vizact
import vizspace
import vizcam
import vizinfo

# import general libraries
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

def plot_image(img_array):
	plt.imshow(img_array, cmap = 'gray')

def rgb2gray(img_rgb):
	R_CONST = 0.299
	G_CONST = 0.587
	B_CONST = 0.114
	img_shape = img_rgb.shape
	M = img_shape[0]
	N = img_shape[1]
	K = img_shape[2]
	img_gray = np.zeros([M,N])
	for m in range(M):
		for n in range(N):
			img_rgb_r = img_rgb[m,n,0]
			img_rgb_g = img_rgb[m,n,1]
			img_rgb_b = img_rgb[m,n,2]

			img_gray[m,n] = (R_CONST * img_rgb_r) + (G_CONST * img_rgb_g) + (B_CONST * img_rgb_b)
	return img_gray

def hist_256(img_array):
	#initialize histogram vector
	hist_vector = np.zeros(256)
	for i in range(img_array.shape[0]):
		for j in range(img_array.shape[1]):
			pixel_val = int(round(img_array[i,j]))
			hist_vector[pixel_val] = hist_vector[pixel_val] + 1
	return hist_vector

def hist_eq(img, img_hist):
	#use cumulative distribution function
	cdf = cumulative_sum(img_hist)
	cdf = 255 * (cdf / cdf[-1])
	#linearly interpolate the cdf for equalized pixel values
	img_equalized = np.interp(img.flatten(), np.arange(256), cdf)
	#reshape to a matrix with dimensions of the original image
	img_equalized = img_equalized.reshape(img.shape)
	return img_equalized, cdf

def cumulative_sum(array):
	""" returns an array """ 
	array_length =  array.shape[0]
	cumulative_sum = np.zeros(array_length)
	for i in range(array_length):
		if i == 0:
			cumulative_sum[i] = array[i]
		else:
			cumulative_sum[i] = cumulative_sum[i-1] + array[i]
	return cumulative_sum
	
def shiftCurves(image, slidervalue):
	grey = rgb2gray(image)
	greyhist = hist_256(grey)
	grey = grey.flatten()
	image_length = grey.shape[0]
	slidervalue = slidervalue/(image_length/2)
	counter = 127
	for i in range(greyhist.shape[0]):
		if slidervalue*i <= slidervalue:
			greyhist[i] += slidervalue*i
		else:
			greyhist[i] += slidervalue*i - slidervalue*(i-counter)
			counter -=1
	return hist_eq(grey, greyhist, 8)
		
if __name__ =="__main__":
	img = load_image('87.png')	# loads image into an array using scipy.misc.imread returning a nth dimensional array
	test = shiftCurves(img,200)	# takes image and turns it grey, generates a histogram, flattens the image array into a 1D array, adds proportionally adds slidervalues to the histogram, then computes the cdf and 'equalizes' the image
	
	# viz specific commands to load image to viz's renderer as a texture on a quad
	# also sets camera field of view and keyboard controls to standard FPS controls
	viz.setMultiSample(4)
	viz.fov(60)
	viz.go()
	viz.cam.setHandler(vizcam.KeyboardCamera())
	picture = viz.addTexture('87.png')
	quad = viz.addTexQuad()
	quad.setPosition([0,1.5,4])
	quad.texture(picture)
	