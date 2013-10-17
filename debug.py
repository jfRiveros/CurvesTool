#Import Modules

# General Modules
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Viz specific Modules
# import viz
# import vizact
# import vizspace
# import vizcam
# import vizinfo

def load_image(file_path):
	img_array = scipy.misc.imread(file_path)
	return img_array

def plt_img(img_array, map='gray',bool=0):
	if bool == 0:
		plt.imshow(img_array,cmap=map)
	else:
		plt.imshow(img_array,cmap=map),plt.show()

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

def powerlaw(img, slidervalue, sigma):
	shiftedImg = np.power(np.round(np.power(img+sigma, slidervalue)),1/slidervalue)
	return shiftedImg

def logT(img):
	loggedImg = np.exp(np.round(np.log(img+1)))
	return loggedImg

if __name__ =="__main__":
	img = load_image('21391920.jpg')	# loads image into an array using scipy.misc.imread returning a nth dimensional array
	grey = rgb2gray(img)
	yolo = powerlaw(grey, 1, 150)
	bolo = logT(grey)
	
	# #matplotlib graphs for testing
	# plt.subplot(121),plt_img(grey)
	# plt.subplot(122),plt_img(yolo)
		#plt.subplot(133),plt_img(bolo)
	#plt.show()

	# viz specific commands to load image to viz's renderer as a texture on a quad
	# also sets camera field of view and keyboard controls to standard FPS controls
	# viz.setMultiSample(4)
	# viz.fov(60)
	# viz.go()
	# viz.cam.setHandler(vizcam.KeyboardCamera())
	# picture = viz.addTexture(test)
	# quad = viz.addTexQuad()
	# quad.setPosition([0,1.5,4])
	# quad.texture(picture)