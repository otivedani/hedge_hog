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

def hog(image2d, cell_size=(8,8), block_size=(2,2), block_stride=(1,1), 
    nbins=9, useSigned=False, useInterpolation=False, normalizeType='L2.old', ravel=True):
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
    csize_y, csize_x = cell_size
    h, w = image2d.shape
    h_incell, w_incell = h//csize_y, w//csize_x
    #lazy cropping
    h, w = h_incell*csize_y, w_incell*csize_x
    image2d = image2d[:h,:w]
    degreebase = 180 if not useSigned else 360
        
    _cell_mapper = np.arange(h_incell*w_incell).reshape(h_incell, w_incell).repeat(csize_x, axis=1).repeat(csize_y, axis=0)
    
    # Section 1. gradient image x and y
    gX = filters.conv_filter(image2d,[1,0,-1],[0,1,0])
    gY = filters.conv_filter(image2d,[0,1,0],[1,0,-1])
    
    # Section 2. gradient magnitude and orientation
    mag, ang = filters.toPolar(gX, gY, signed=useSigned)

    # Section 3. trilinear interpolation voting
    if useInterpolation:
        
        magcoef, angcoef = linterp(ang, nbins, useSigned)
        ang2x = ((nbins-1)*angcoef/degreebase).astype('int')

        _cell_mapper_xtra = np.pad(_cell_mapper, ((csize_y//2, csize_y//2),(csize_x//2, csize_x//2)), 'edge')
        # print(np.all(_place_to_bin[4:-4,4:-4]==_cell_mapper) #must be True)
        
        _pibx = np.array((
            _cell_mapper_xtra[:h,:w], \
            _cell_mapper_xtra[:h,-w:], \
            _cell_mapper_xtra[-h:,:w], \
            _cell_mapper_xtra[-h:,-w:]))

        # # !!!important, if you dont want data loss (Python 2.x) -> mag2x = mag2x.astype(long)
        magx = blinterp(magcoef*mag[None,:,:],cell_size)
        binx = ang2x[:,None,:,:]+(_pibx[None,:,:,:]*nbins)

    else:
        _bin_step = (degreebase/(nbins-1))
        angx = ((ang+_bin_step/2)//_bin_step).astype('int')
        _pibx = _cell_mapper
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

    #L2-norm
    if normalizeType=='L2':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2)+(1e-7)**2)[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #L2-hys
    elif normalizeType=='L2-hys':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2)+(1e-7)**2)[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
        blockhists[np.where(blockhists>0.2)] = 0.2
        bmax, bmin = np.amax(blockhists, axis=-1), np.amin(blockhists, axis=-1)
        blockhists = (blockhists - bmin[:,:,None])/np.where(bmax==bmin, 1, (bmax - bmin))[:,:,None]
    #L1-norm
    elif normalizeType=='L1':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #L1-sqrt
    elif normalizeType=='L1-sqrt':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
        blockhists = np.sqrt(blockhists)
    #make this default normalize method
    elif normalizeType=='L2.old':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2))[:,:,None]
        blockhists /= np.where(divisor!=0, divisor, 1)
    #not normalized, no good. debug purpose only.
    else:
        pass
        
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
    x = ang    
    x_1 = (ang//binStep)*binStep
    c = (x-x_1)/binStep
    x_2 = np.where(x_1 < degreebase, x_1+binStep, 0)
    # return np.array([[c*mag, x_2],[(1-c)*mag, x_1]])
    x_1 = np.where(x_1 < degreebase, x_1, 0)
    return np.array([c,(1-c)]), \
            np.array([x_2, x_1])
    
def blinterp(mag, cellSize):
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

    xi_i = ((xi+cellSize[0]/2)//cellSize[0])*cellSize[0]
    yj_j = ((yj+cellSize[1]/2)//cellSize[1])*cellSize[1]
    x_coef = ((xi+cellSize[0]/2)-xi_i)/cellSize[0]
    y_coef = ((yj+cellSize[1]/2)-yj_j)/cellSize[1]

    _blin_coefs = np.asarray(\
    [  (1-x_coef)*(1-y_coef),\
        x_coef*(1-y_coef),\
        (1-x_coef)*y_coef,\
        x_coef*y_coef           ])
    
    """
    mag.shape = (..., height, width)
    blincoefs.shape = (4, height, width)
    desired output = (..., 4, height, width)
    howto : (..., height, width) -> (..., height*width) ->
                (..., 4 ,height, width)
        using 4 to avoid confusion
    """
    return mag[:, None, :, :]*_blin_coefs
