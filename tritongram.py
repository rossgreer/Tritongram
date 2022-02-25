"""
Tritongram

Digital Image Processing Hall of Fame
Filters designed by UCSD students of ECE 253 and ECE 172A

For use by UCSD ECE 253 and 172A students only, 
in accordance with UCSD Academic Integrity policy. 

Script by regreer@ucsd.edu
"""

import cv2
import numpy as np
import matplotlib as mpl
import warnings
from matplotlib import pyplot as plt


def daydream(im,colormap='jet'):
	'''
	This filter will take a single RGB image and compute a rough segmentation 
	(up to 10 different colors) of the original image based on colors
	in the given colormap with a white outline that preserves some details of the original pictures.
	
	Input: 
	im: MxNx3 ndarray, it should be a 3-channel RGB image.
	colormap: str, the name of the standard colormap provided in matplotlib.
	
	Output:
	out: MxNx3 ndarray, the output image.
	
	Author: Feng Li 
	'''
	from skimage import morphology as morph
	from skimage import exposure
	from skimage import filters
	from skimage import feature
	im=np.asarray(im)
	dim=im.shape
	
	if len(dim)==2:
		colormap='Greys'
		im_bw=im.copy()
		im=np.dstack([im,im,im]).astype(np.uint8)
		warnings.warn('Input might be a 2d greyscale image. It would be converted into RGB image by assigning same intensity values for R,G,B channels'+ 
			  ' which might results in errors. Please ensure the input image is a single 3-channel RGB image.')
	elif (len(dim)==3) and (dim[2]>3):
		warnings.warn('Input image has more than 3 channels. The first 3 channels would be used for the computation'+
					  ' which might results in errors. Please ensure the input image is a single RGB image.')
		im_bw=np.average(im,axis=2,weights=[0.299,0.587,0.114])
	elif (len(dim)==3) and (dim[2]==3):
		im_bw=np.average(im,axis=2,weights=[0.299,0.587,0.114])
	else:
		raise Exception('please check the format of the input image')
		
	N=dim[0]*dim[1]
	#performing canny edge detection with a low-pass gaussian filter with sigma=2.
	edge5=feature.canny(im_bw,sigma=2,low_threshold=0.5,high_threshold=0.9,use_quantiles=True).astype(int)
	edge=morph.binary_dilation(edge5,morph.disk(1)) #thicken the edges by dilation

	#create a rectangular foot print for the median filter
	footp=morph.rectangle(int(dim[0]/40),int(dim[1]/40))
	
	#perform a median filter on the original image to downsample the color values in each channel.
	R=filters.rank.median(im[:,:,0],footp) 
	G=filters.rank.median(im[:,:,1],footp)
	B=filters.rank.median(im[:,:,2],footp)
	medim=(np.dstack([R,G,B])).astype(int) #image after median filter on each color channel
	cmap=mpl.cm.get_cmap(colormap,10) #import the given colormap, only ten color values would be used for cleaner result
	cols=cmap(np.arange(0,10,1))[:,:3].T*255 #each row contains the color list for each channel of R,G,B.

	def find_palette(cvector,Qcvector):
		'''
		find the closest RGB color from the given list of color vectors.
		
		input:
		cvector: (3, ) 1-d array. The color vector of the input pixel
		Qcvector: 3xN ndarray. The reference set of color vectors.
		
		output:
		newcvector: (3,) 1-d array. The closest color vector from the reference set.
		'''
		cvector=cvector.astype(float)
		[Rq,Gq,Bq]=Qcvector #given reference color
		newcvector=np.zeros(cvector.shape)
		newcvector[0]=Rq[np.argmin(np.abs(cvector[0]-Rq))]
		newcvector[1]=Gq[np.argmin(np.abs(cvector[1]-Gq))]
		newcvector[2]=Bq[np.argmin(np.abs(cvector[2]-Bq))]
		return newcvector

	newim=np.zeros_like(im)
	
	for i in range(dim[0]):
		for j in range(dim[1]):
			old_pixel=medim[i,j,:].copy()
			#replacing the pixel color vector with the closest color vector in the reference set given by the colormap
			new_pixel=find_palette(old_pixel,cols)
			newim[i,j,:]=new_pixel

	out=newim.astype(int)
	out[np.dstack([edge,edge,edge])]=220 #add a bright edge to the color segmented image.
	
	return out

def colormosaic(im, im_ref=None, mosaic=5, sharp=False):
	'''
	This filter will take a single RGB image and perform a dithering to get
	mosaic
		
	Input: 
	im: MxNx3 ndarray, it should be a 3-channel RGB image.
	im_ref: MxNx3 ndarray, it should be a 3-channel RGB image.
	mosaic: int, should be greater than zero
	sharp: boolean, default False
	
	Output:
	out: MxNx3 ndarray, the output image.
	
	Author: Meng Dong
	'''
	def stats(im):
		l, a, b = cv2.split(im)
		mean_l, std_l = l.mean(), l.std()
		mean_a, std_a = a.mean(), a.std()
		mean_b, std_b = b.mean(), b.std()
		return mean_l, std_l, mean_a, std_a, mean_b, std_b

	height, width = im.shape[:2]
	im_new = im.copy()
	if mosaic > 0:
		for i in range(height):
			for j in range(width):
				i_dither = i + np.random.randint(-mosaic, mosaic)
				j_dither = j + np.random.randint(-mosaic, mosaic)
				i_new = max(0, min(height-1, i_dither))
				j_new = max(0, min(width-1, j_dither))
				im_new[i, j, :] = im[i_new, j_new, :]
	if sharp:
		kernel = np.array([[0, -1, 0],
						   [-1, 5,-1],
						   [0, -1, 0]])
		im_new = cv2.filter2D(im_new, -1, kernel, borderType=4)
	if im_ref is not None:
		source = cv2.cvtColor(im_ref, cv2.COLOR_BGR2LAB).astype("float32")
		target = cv2.cvtColor(im_new, cv2.COLOR_BGR2LAB).astype("float32")
		stats_source = stats(source)
		stats_target = stats(target)

		l_t, a_t, b_t = cv2.split(target)
		l_t = (l_t - stats_target[0]) * stats_target[1] / stats_source[1] + stats_source[0]
		a_t = (a_t - stats_target[2]) * stats_target[3] / stats_source[3] + stats_source[2]
		b_t = (b_t - stats_target[4]) * stats_target[5] / stats_source[5] + stats_source[4]
		l_t = np.clip(l_t, 0, 255)
		a_t = np.clip(a_t, 0, 255)
		b_t = np.clip(b_t, 0, 255)
		im_new = cv2.merge([l_t, a_t, b_t])
		im_new = cv2.cvtColor(im_new.astype("uint8"), cv2.COLOR_LAB2BGR)
	return im_new

def disco_filter(rgb_img, ksize=5):
	''' 
	Function that colors the image based on the orientation and magnitude of the edge derivatives. 
	Input is a single RGB image and the blurring kernel size 
	Output is an recolored RGB image

	Input: 
	rgb_img: numpy array, RGB image
	ksize: int, kernel size

	Output:
	out: numpy array, RGB image

	Author: Leeor Nehardea
	'''

	import random 
	import copy

	# Change to gray scale
	gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
	
	# Blur the image
	assert ksize > 0, "Kernel size must be larger than 0"
	gray = cv2.blur(gray, (ksize, ksize))
	
	# Sobel derivatives on X&Y directions
	derivX = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
	derivY = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
	
	# Orientation and magnitude
	orien = cv2.phase(derivX, derivY, angleInDegrees=True)
	mag = cv2.magnitude(derivX, derivY)
	
	# Threshold
	ret, mask = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
	
	# Colors option
	salmon = np.array([225,139,107])
	red = np.array([0,0,255])
	lilac = np.array([74,47,146])
	cyan = np.array([255,255,0])
	green = np.array([0,255,0])
	yellow = np.array([0,255,255])
	sunny = np.array([255,247,122])
	blue = np.array([132,200,226])
	white = np.array([255,255,255])
	
	# New image 
	out = np.zeros((orien.shape[0], orien.shape[1], 3), dtype=np.uint8)
	
	# List of the colors and angles 
	colors = [salmon, red, lilac, cyan, green, yellow, sunny, blue, white]
	angles = [0,45,90,135,180,225,270,315]
	
	# Color the image (randomly) based on the orientation value
	for angle in angles:
		# Choose a color at random
		rand_color = random.randint(1,len(colors)) - 1
		# Select range values to color the image
		bottom = angle
		top = angle + 45
		# Color the image
		out[(mask == 255) & (orien >= bottom) & (orien < top)] = colors[rand_color]
		
		# Delete the color so it won't be used again
		del colors[rand_color]
		
	return out

def dither_watermark(img, watermark, mode='b'):
	"""
	Dithering to add a watermark

	Input 
	img: numpy array, RGB image
	watermark:  numpy array, RGB image (dims smaller than image)
	mode: str, 'b' or 'd'
	'b' means adaptive threshold and 'd' means Floyd_Steinberg dithering.

	Returns:
	out: RGB image

	Author: Haochen Zhang
	"""
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt

	def find_closest_palette_color(value, level=2):
		# generate valid values
		thresholds = np.linspace(0,255,level)
		valid_values = np.round(thresholds)
		#print(valid_values)
		# find nearest valid value
		tmp = np.abs(valid_values-value)
		min_tmp = np.min(tmp)
		min_loc = np.where(tmp == min_tmp)
		return valid_values[min_loc][0].astype(np.uint8)

	def Floyd_Steinberg(img):
		H,W = img.shape
		im = img.copy()
		im = im.astype(np.float32)
		quant_errors = im.astype(np.float32)
		for y in range(H):
			for x in range(W):
				old_value = im[y,x]
				new_value = find_closest_palette_color(old_value)
				#print(old_value,new_value)
				im[y,x] = new_value
				quant_error=old_value-new_value
				if x+1 < W:
					im[y, x + 1] = im[y,x + 1] + quant_error * 7. / 16.
				if y+1 < H:
					im[y + 1 ,x]=im[y + 1,x] + quant_error * 5. / 16.
					if x-1 >=0:
						im[y + 1,x - 1]=im[y + 1,x - 1] + quant_error * 3. /16.
					if x+1 <W:
						im[y + 1,x + 1]=im[y + 1,x + 1] + quant_error * 1. /16.
		im = np.clip(np.round(im),0,255)
		im = im.astype(np.uint8)
		return(im)

	def my_fun_add_watermark(img, watermark=None, mode='b'):
		'''
		img: the image on which you want to add watermark. Either 3 channel RGB image or 2 channel gray image is OK;
		watermark: 
		You can give your own watermark. Ideally, it should be a binary image. Otherwise, the algorithm will make some changes:
		If watermark is gray image, it would be binarized using binarization or dithering.
		If watermark is RGB, it would be convert to gray and then apply binarization or dithering.
		mode: 'b' or 'd'. This parameter decides the binarization method:
		'b' means adaptive threshold and 'd' means Floyd_Steinberg dithering.
		'''
		# check input
		if len(img.shape) > 2:
			if np.sum(np.abs(img[:,:,0]-img[:,:,1]))==0 and np.sum(np.abs(img[:,:,0]-img[:,:,2]))==0:
				input_type = 2 # 3 channel gray
				img_y = img[:,:,0]
			else:
				input_type = 3 # 3 channel RGB
				img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
				img_y = img_ycbcr[:,:,0]
		else:
			input_type = 1 # 2 channel gray
			img_y = img
		hei,wid = img_y.shape

		# check watermark
		if not type(watermark) is np.ndarray:
			watermark = cv2.imread('./wmBinary.png') #BGR
			watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB) #RGB
		# check gray
		if len(watermark.shape) > 2:
			watermark = cv2.cvtColor(watermark, cv2.COLOR_RGB2GRAY)

		# check binary
		tmp = watermark.copy()
		watermark = cv2.resize(watermark, (wid,hei), interpolation = cv2.INTER_CUBIC)
		tmp[tmp==255] = 0
		if np.max(tmp) > 0:
			if mode == 'd':
				watermark = Floyd_Steinberg(watermark)
			else:
				watermark = cv2.adaptiveThreshold(watermark,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
		else:
			watermark[watermark>127.5] = 255
			watermark[watermark<127.5] = 0
		plt.figure()
		
		# add watermark
		watermark[watermark>0]=1
		tmp = np.ones((hei,wid),dtype=np.uint8)*254
		img_H7 = cv2.bitwise_and(img_y,tmp)
		img_y = cv2.bitwise_or(img_H7,watermark)
		if input_type==3:
			img_ycbcr[:,:,0] = img_y
			img = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2RGB)
		elif input_type==2:
			img = np.stack([img_y,img_y,img_y],axis=2)
		else:
			img = img_y
		return img

	# extract watermark
	def my_fun_extract_watermark(img):
		# check input
		if len(img.shape) > 2:
			if np.sum(np.abs(img[:,:,0]-img[:,:,1]))==0 and np.sum(np.abs(img[:,:,0]-img[:,:,2]))==0:
				input_type = 2 # 3 channel gray
				img_y = img[:,:,0]
			else:
				input_type = 3 # 3 channel RGB
				img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
				img_y = img_ycbcr[:,:,0]
		else:
			input_type = 1 # 2 channel gray
			img_y = img
		hei,wid = img_y.shape
		tmp = np.ones((hei,wid),dtype=np.uint8)
		watermark = cv2.bitwise_and(img_y,tmp)
		watermark[watermark>0]=255
		return(np.stack([watermark,watermark,watermark],axis=2))

	out = my_fun_add_watermark(img, watermark, mode)
	return out

def triton_filter(im):
	"""
	Takes an input image im and returns a color segmented version

	Input: im, numpy array, RGB image 
	Output: im_seq, numpy array, RGB image

	Author: Manas Bedmutha
	"""

	
	# Initialize segmentation params
	M = 3              # For R,G,B channels
	nclusters = 9      # Number of k-means clusters
	max_iter = 15      # Max number of iterations if k-means doesn't converge
	
	# Create features and initial cluster centers
	features = im.reshape(im.size//M,M)*1.0
	centers = features[np.random.randint(0, len(features),nclusters)]
	
	# Variables for better computations
	N = features.shape[0]
	k = centers.shape[0]
	dist = np.zeros((N,k))

	# Initial conditions
	last_centers = centers
	thresh = 0.5
	epochs = 0

	# K-means iterations
	while ((np.linalg.norm(centers - last_centers) > thresh) or (epochs==0)) and (epochs < max_iter):
		
		# Calculate distance from each cluster center
		for i in range(k):
			dist[:,i] = np.linalg.norm(features - centers[i], axis=-1)
			
		# Update new classes and centers 
		newClass = np.argmin(dist, axis=1)

		last_centers = centers
		for i in range(k):
			centers[i] = features[np.where(newClass == i)].mean(axis=0)

		epochs += 1
	
	# Replace all pixels with current shades
	im_seq = centers[newClass.reshape(im.shape[:2])]
	# Scale values to wrap around, just to spice up the shades
	im_seq = np.uint8(im_seq*1.5)
	
	# Return outputs
	return im_seq

filter_list = [daydream, colormosaic, disco_filter, dither_watermark, triton_filter]
n = len(filter_list)+1

print("Welcome to Tritongram, the premier image processing filter suite by UCSD students.")
path = input("Please provide the path to the image you would like to process:")

# Gather image input
img = cv2.imread(path)

idx = int(input("Enter a filter number (1 through "+str(n)+") to view:")) - 1

# Receive filtered out
out = filter_list[idx](img)

# Show output
plt.imshow(out)
plt.show()