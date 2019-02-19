#!/usr/bin/env python2
"""
    hog.py
    
    Histogram of Oriented Gradients 
    in Python numpy
    (for educational purpose only)
    credit to : Dalal & Triggs 

"""
#Author @otivedani 


import numpy as np
from ..improc import filters
from ..powerup import indextra

#TODO adapting those parameter from opencv2
# useSignedGradients = False
# #unknownvariable
# derivAperture = 1
# winSigma = -1.
# histogramNormType = 0
# L2HysThreshold = 0.2
# gammaCorrection = 1
# nlevels = 64


def hog(image2d, cell_size=(8,8), block_size=(2,2), block_stride=(1,1), nbins=9, useSigned=False, useInterpolation=False, normalizeType='L2.old', ravel=True):
    """
    Parameters : 
	------------
        image2d - [n=2]darray, one channel image
        cell_size - (int, int), size in pixel-per-cell
        block_size - (int, int), size in cell-per-block 
        block_stride - (int, int), amount to stride per-block
        nbins - int, size of bin
        useSigned - 0-360 is True 
                    0-180 is False (default)
        useInterpolation - using linear-bilinear interpolation is True, not using is False (default)
                        
        ...(tba)
    
    Return :
    --------
    normalized block vector, flattened

    """

    # Section 0. Precompute variables
    h, w = image2d.shape
    h_incell, w_incell = h//cell_size[0], w//cell_size[1]
    #lazy cropping
    h, w = h_incell*cell_size[0], w_incell*cell_size[1]
    image2d = image2d[:h,:w]

    degreebase = 180 if not useSigned else 360
        
    _dum = np.arange(h_incell*w_incell).reshape(h_incell,w_incell).repeat(cell_size[0], axis=1).repeat(cell_size[1], axis=0)
    _place_in_bin = np.pad(_dum, (cell_size[0]//2,cell_size[1]//2), 'reflect')
    # print(np.all(_place_to_bin[4:-4,4:-4]==_dum) #must be True)
    
    # Section 1. gradient image x and y
    gX = filters.conv_filter(image2d,[1,0,-1],[0,1,0])
    gY = filters.conv_filter(image2d,[0,1,0],[1,0,-1])
    
    # Section 2. gradient magnitude and orientation
    mag, ang = filters.toPolar(gX, gY, signed=useSigned)

    # Section 3. trilinear interpolation voting
    if useInterpolation:
        
        magcoef, angcoef = linterp(ang, nbins, useSigned)
        ang2x = ((nbins-1)*angcoef/degreebase).astype('int')

        dum = np.pad(_dum, (cell_size[0]//2,cell_size[1]//2), 'reflect')
        _pibx = np.array((_place_in_bin[:-8,:-8], _place_in_bin[:-8,8:], _place_in_bin[8:,:-8], _place_in_bin[8:,8:]))
        # _pibx = np.array((_place_in_bin[8:,8:],_place_in_bin[8:,:-8], _place_in_bin[:-8,8:], _place_in_bin[:-8,:-8]))
        # _pibx = np.array([_dum,_dum,_dum,_dum])

        # # !!!important, if you dont want data loss
        # mag2x = mag2x.astype(long)
        magx = blinterp(magcoef*mag[None,:,:],cell_size)
        binx = ang2x[:,None,:,:]+(_pibx*nbins)[None,:,:,:]

        # print(mag2x.shape, ang2x.shape, bin2x.shape, newbin2x.shape, (lemper1*_hog_param_visd['orientations']).shape)

    else:
        _bin_step = (degreebase/(nbins-1))
        angx = ((ang+_bin_step/2)//_bin_step).astype('int')
        _pibx = _place_in_bin[cell_size[0]//2:-cell_size[0]//2,cell_size[1]//2:-cell_size[1]//2]
        binx = angx[None, :,:]+(_pibx[None, :,:]*nbins)
        magx = mag[None, :, :]
        

    hists = np.bincount(binx.ravel(), magx.ravel(), h_incell*w_incell*nbins)\
                .reshape((h_incell,w_incell,nbins))
    
    # Section 4. block normalization
    _bindex = np.arange(hists.size).reshape(hists.shape[0],-1)

    _cpb = block_size[0], block_size[1]*nbins
    _blockix = indextra.convolver(_bindex, _cpb, (block_stride[0],block_stride[1]*nbins))
    _blockix = _blockix.reshape(_blockix.shape[0], _blockix.shape[1], _cpb[0]*_cpb[1])

    blockhists = hists.ravel()[_blockix]

    if normalizeType=='L2.old':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #L2-norm
    elif normalizeType=='L2':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2)+(1e-7)**2)[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #L1-norm
    elif normalizeType=='L1':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #L1-sqrt
    elif normalizeType=='L1-sqrt':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
        blockhists = np.sqrt(blockhists)
        
    if ravel:
        return blockhists.ravel()
    else:
        return blockhists

def linterp(ang, nbins, signed=False):
    """
    Parameters : 
	------------
        ang: numpy 2darray - orientation coordinate
        nbins: integer - divisor of angle
        signed: bool - orientation range 
                        (0,360) if True, 
                        (0,180) if False (default)
    Return :
    --------
        tuples of numpy 2darray magnitude and angle
        [c  ,               (1-c)],
        [x_2,                 x_1]
    """
    degreebase = 180 if not signed else 360

    #coefficient
    binStep = degreebase/nbins
    x = np.copy(ang)
    x_1 = np.floor(ang/binStep)*binStep
    c = (x-x_1)/binStep
    x_2 = np.where(x_1 < 180, x_1+binStep, x_1)

    # return np.array([[c*mag, x_2],[(1-c)*mag, x_1]])
    return np.array([c,(1-c)]), \
            np.array([x_2, x_1])
    
def blinterp(mag, cellSize):
    #FLAG
    """
    Parameters : 
	------------
        mag: numpy 2darray - magnitude weights
        cellSize: (int,int) - cell size to determine ranges
        
    Return :
    --------
        magnitudes in their respective bilinear position
        
        (mag_00,mag_01,mag_10,mag_11)

        where :
        mag = mag_00 + mag_01 + mag_10 + mag_11
    """
    xi = np.tile(np.arange(mag.shape[-1]), (mag.shape[-2],1))
    yj = np.tile(np.arange(mag.shape[-2]), (mag.shape[-1],1)).T

    xi_i = np.floor((xi+cellSize[0]/2)/cellSize[0])*cellSize[0]
    yj_j = np.floor((yj+cellSize[1]/2)/cellSize[1])*cellSize[1]
    x_coef = ((xi+cellSize[0]/2)-xi_i)/cellSize[0]
    y_coef = ((yj+cellSize[1]/2)-yj_j)/cellSize[1]

    _blin_coefs = np.asarray(\
    [  (1-x_coef)*(1-y_coef),\
        x_coef*(1-y_coef),\
        (1-x_coef)*y_coef,\
        x_coef*y_coef           ])
    #
    # mag[0,0,0] = 0.1234
    # mag[1,0,0] = 5.6789
    # print(_blin_coefs.shape, mag.shape)
#     print(np.all(np.sum(_blin_coefs, axis=(0))==1))
#     print(np.amax(_blin_coefs), np.amin(_blin_coefs), _blin_coefs.dtype)
    
    """
    mag.shape = (..., height, width)
    blincoefs.shape = (4, height, width)
    desired output = (..., 4, height, width)
    howto : (..., height, width) -> (..., height*width) ->
                (..., 4 ,height, width)
        using 4 to avoid confusion
    """
    return mag[:, None, :, :]*_blin_coefs
