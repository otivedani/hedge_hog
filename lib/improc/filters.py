#!/usr/bin/env python2
"""
filters.py - various image filtering operation

"""
#Author @otivedani 

import numpy as np

# ::
def sobelX_HC(img):
    """
    Parameters : 
	------------
        img: numpy 2darray
    """
    img2 = np.pad(img,((1,1),(1,1)),'constant', constant_values=((0, 0),(255, 0)))
    result_h = img2[:,2:] - img2[:,:-2]
    result_v = result_h[2:] + result_h[1:-1] + result_h[:-2]
    # result_v[1:-1]
    return np.clip(result_v,0,255)

def sobelY_HC(img):
    """
    Parameters : 
	------------
        img: numpy 2darray
    """
    img2 = np.pad(img,((1,1),(1,1)),'constant', constant_values=((0, 255),(0, 0)))
    # print img2
    result_h = img2[:,2:] + img2[:,1:-1] + img2[:,:-2]
    # print result_h
    result_v = result_h[2:] - result_h[:-2]
    # print result_v
    # print len(result_v)
    return np.clip(result_v,0,255)

# Section 1: numpy-based 2D convolutional filter, improved version from sobelY_HC and sobelX_HC
def conv_filter(img, h_kernel, v_kernel, clip=False):
    """
    Convolve 2d array horizontally then vertically.
    
    Parameters : 
	------------
        img: numpy 2darray
        h_kernel: 1darray
        v_kernel: 1darray
        clip: if true - normalize to range(0,255)

    Return :
    --------
    img · (h·v)
    
    """
    h_klen = len(h_kernel)
    v_klen = len(v_kernel)

    h_pad = (h_klen-1)/2
    v_pad = (v_klen-1)/2

    result_h = np.zeros((img.shape[0]+(2*h_pad)+2,img.shape[1]))
    result_v = np.zeros(img.shape)

    # img2 = np.pad(img,((1+(h_pad),1+(h_pad)),(1+(v_pad),1+(v_pad))),'constant', constant_values=((0, 0),(0, 0)))
    img2 = np.pad(img,((1+(h_pad),1+(h_pad)),(1+(v_pad),1+(v_pad))),'reflect')

    #work like matlab.conv2(u,v,A)
    #TODO kill looping (vectorize)
    for i in range(h_klen):
        # result_h += h_kernel[i]*img2[:,1+i:len(img)+1+i]
        result_h += h_kernel[i]*img2[:,-(len(img[0])+1+i):-(1+i)]
    for j in range(v_klen):
        # result_v += v_kernel[j]*result_h[1+j:len(img)+1+j]
        result_v += v_kernel[j]*result_h[-(len(img)+1+j):-(1+j)]

    if clip:
        return np.clip(result_v,0,255)
    else:
        return result_v

#-- end of Section 1


# Section 2 : Magnitude, Angle vector calculation
def toPolar(gX, gY, signed=False, dtype='float'):
    """
    Parameters : 
	------------
        gX: numpy 2darray - cartesian coordinate
        gY: numpy 2darray - cartesian coordinate
        signed: if true - change to range(0,360) == similar to 'cv2'
                if false - change to range(0,180)(default)
    Return :
    --------
    polar coordinate (ω,θ)
    """

    _basis = 360 if signed else 180

    # still return (-180,180)
    _ang = np.degrees(np.arctan2(gY,gX))

    mag = np.sqrt((np.square(gX))+(np.square(gY)))
    ang = np.where(_ang < 0, _ang + _basis, _ang)

    return mag.astype(dtype), ang.astype(dtype)
#-- end of Section 2

# Section 3 : x-position, y-position vector calculation
def toCart(ang, mag, signed=False, dtype='float'):
    """
    Parameters : 
	------------
        ang: array-like numpy, in degrees
        mag: array-like numpy
        signed: if true - change to range(0,360)
                if false - change to range(0,180)

    Return :
    --------
    cartesian coordinate (x,y)
    """

    _basis = 360 if signed else 180

    x = mag * np.cos(ang*np.pi/_basis)
    y = mag * np.sin(ang*np.pi/_basis)

    return x.astype(dtype), y.astype(dtype)
#-- end of Section 3

# Section 4 :
# TBA
#-- end of Section 4
