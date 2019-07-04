"""
filters.py - various image filtering operation

"""
#Author @otivedani 

import numpy as np

# Section 0: 2d edge convolution 
def simple_edge_gradients(img):
    """
    Simplifying convolution of simple edge kernel [-1,0,1] & [-1,0,1].T 

    Parameters : 
	------------
        img     : numpy 2darray
    Return :
    --------
        gX, gY  : edge gradients
    """
    gmi = np.pad(img, (1,1), 'edge').astype(np.float_)
    gX = (-gmi[:,:-2] + gmi[:,2:])[1:-1,:]
    gY = (-gmi[:-2,:] + gmi[2:,:])[:,1:-1]
    return gX, gY

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
    img (dot) (h(dot)v)
    
    """
    h_klen = len(h_kernel)
    v_klen = len(v_kernel)

    h_pad = (h_klen-1)//2
    v_pad = (v_klen-1)//2
    
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
def toPolar(gX, gY, signed=False):
    """
    Parameters : 
	------------
        gX: numpy array - X axis cartesian coordinate
        gY: numpy array - Y axis cartesian coordinate
        signed: if True - change to range(0,360)(similar to 'cv2')
                if False - change to range(0,180)(default)
    Return :
    --------
        magnitude, theta: polar coordinate
    """

    _basis = 360 if signed else 180

    mag = np.sqrt((np.square(gX))+(np.square(gY)))
    ori = np.degrees(np.arctan2(gY,gX))
    ori[ ori < 0 ] += _basis

    return mag, ori
#-- end of Section 2

# Section 3 : x-position, y-position vector calculation
def toCart(ori, mag, signed=False, dtype='float'):
    """
    Parameters : 
	------------
        ori: array-like numpy, in degrees
        mag: array-like numpy
        signed: if true - change to range(0,360)
                if false - change to range(0,180)

    Return :
    --------
    cartesian coordinate (x,y)
    """

    _basis = 360 if signed else 180

    x = mag * np.cos(ori*np.pi/_basis)
    y = mag * np.sin(ori*np.pi/_basis)

    return x.astype(dtype), y.astype(dtype)
#-- end of Section 3

# Section 4 :
# TBA
#-- end of Section 4
