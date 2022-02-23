"""
Tritongram

Digital Image Processing Hall of Fame
Filters designed by UCSD students of ECE 253 and ECE 172A

For use by UCSD ECE 253 and 172A students only, 
in accordance with UCSD Academic Integrity policy. 

Script by regreer@ucsd.edu
"""

from skimage import morphology as morph
from skimage import exposure
from skimage import filters
import numpy as np
import matplotlib as mpl
import warnings



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
            new_pixel=find_palette(old_pixel,cols) #replacing the pixel color vector with the closest color vector in the reference set given by the colormap
            newim[i,j,:]=new_pixel

    out=newim.astype(int)
    out[np.dstack([edge,edge,edge])]=220 #add a bright edge to the color segmented image.
    
    return out



filter_list = [daydream, ]

print("Welcome to Tritongram, the premier image processing filter suite by UCSD students.")
print("Please provide the path to the image you would like to process:")
# Gather image input

print("Enter a filter number (1 through "+str(n)+") to view:")

