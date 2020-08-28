"""
    HOG.py
    
    Histogram of Oriented Gradients 
    in Python numpy
    (for educational purpose only)
    credit to : Dalal & Triggs (2005)

"""
#Author @otivedani 

import numpy as np
from ..improc import filters
from ..powerup import indextra

def hog(image2d, cell_size=(8,8), block_size=(2,2), block_stride=(1,1), 
    nbins=9, useSigned=False, useInterpolation=False, normalizeType='L2', ravel=True):
    """
    Parameters : 
	------------
        image2d             : 2darray
                            one channel image
        cell_size           : (int, int)
                            pixel-per-cell, height, width of one cell in pixels (default (8,8))
        block_size          : (int, int)
                            cell-per-block, height, width of one block in cells (default (2,2))
        block_stride        : (int, int)
                            stride step of the block in cells (default (1,1))
        nbins               : int, 
                            angle bins in one cell (default 9)
        useSigned           : boolean, 
                            True for (0,360)
                            False for (0,180) (default)
        useInterpolation    : boolean,
                            True for using linear-bilinear interpolation voting,
                            False for naive voting
        normalizeType       : 'L2' (default), 'L2-hys', 'L1', 'L1-sqrt'
                            normalization method in blocks
        ravel               : boolean
                            True for return flattened array vector (default)
                            False for return retaining original shape
        
    Return :
    --------
        block normalized vector of Histogram of Oriented Gradients
    """

    # Parameter checking
    if len(image2d.shape) is not 2:
        raise ValueError("Image must be in 2-dimension.")
    if normalizeType not in {'L2', 'L2-hys', 'L1', 'L1-sqrt'}:
        raise ValueError("Norm type not supported.")

    ## Section 0. Precompute variables
    degreebase = 180 if not useSigned else 360

    csize_y, csize_x = cell_size
    h, w = image2d.shape
    h_incell, w_incell = h//csize_y, w//csize_x
    
    # crop image
    if (h%csize_y != 0) or (w%csize_x != 0):
        h, w = h_incell*csize_y, w_incell*csize_x
        image2d = image2d[:h,:w]

    ## Section 1. gradient image x and y
    gX, gY = filters.simple_edge_gradients(image2d)
    
    ## Section 2. gradient magnitude and orientation
    mag, ori = filters.toPolar(gX, gY, signed=useSigned)

    ## Section 3. trilinear interpolation voting
    # bin step size
    binStep = degreebase/nbins

    if useInterpolation:
        
        coef_2, bin_2 = linterp(ori, nbins, binStep)
        coef_4, cmap_4 = blinterp(image2d.shape, cell_size)
        
        magx = mag[None,None,:,:]*coef_2[:,None,:,:]*coef_4[None,:,:,:]
        binx = bin_2[:,None,:,:]+(cmap_4[None,:,:,:]*nbins)

    else:
        # build bin position
        bin_1 = ((ori+binStep/2)//binStep).astype(np.int_)
        # max is nbins-1
        bin_1[bin_1 >= nbins] = 0
        
        # build cell mapper
        xi, yj = np.meshgrid(np.arange(w),np.arange(h))
        cell_mapper = xi//csize_x + (yj//csize_y) * w_incell
        cmap_1 = cell_mapper

        magx = mag[None,:,:]
        binx = bin_1[None,:,:]+(cmap_1[None,:,:]*nbins)
        
    hists = np.bincount(binx.ravel(), magx.ravel(), h_incell*w_incell*nbins)\
                .reshape((h_incell,w_incell,nbins))
    
    ## Section 4. block normalization
    _bindex = np.arange(hists.size).reshape(hists.shape[0],-1)

    _cpb = block_size[0], block_size[1]*nbins
    _blockix = indextra.convolver(_bindex, _cpb, (block_stride[0],block_stride[1]*nbins))
    _blockix = _blockix.reshape(_blockix.shape[0], _blockix.shape[1], _cpb[0]*_cpb[1])

    blockhists = hists.ravel()[_blockix]

    if normalizeType=='L2':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2)+(1e-7)**2)[:,:,None]
        blockhists /= divisor
    elif normalizeType=='L2-hys':
        divisor = np.sqrt(np.sum(blockhists.copy()**2, axis=2)+(1e-7)**2)[:,:,None]
        blockhists /= divisor
        blockhists[blockhists>0.2] = 0.2
        bmax, bmin = np.amax(blockhists, axis=-1), np.amin(blockhists, axis=-1)
        blockhists = (blockhists - bmin[:,:,None])/np.where(bmax==bmin, 1, (bmax - bmin))[:,:,None]
    elif normalizeType=='L1':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= divisor
    elif normalizeType=='L1-sqrt':
        divisor = np.abs(np.sum(blockhists.copy(), axis=2)+(1e-7))[:,:,None]
        blockhists /= divisor
        blockhists = np.sqrt(blockhists)
    else:
        pass
    
    ## Last : HOG feature
    if ravel:
        return blockhists.ravel()
    else:
        return blockhists

def linterp(ori, nbins, binStep):
    """
    Parameters : 
	------------
        ori     : numpy 2darray - orientation coordinate
        nbins   : integer - divisor of angle
        signed  : bool - orientation range 
                True for (0,360) , 
                False for (0,180) (default)
    Return :
    --------
        tuples of numpy 2darray magnitude and angle
        [c  ,  (1-c)],
        [x_2,    x_1]
    """
    
    x = ori/binStep
    x_1 = np.floor(x)
    #coefficient
    c = (x-x_1)
    
    coef = np.asarray([c,(1-c)])
    bpos = np.asarray([x_1+1, x_1], dtype=np.int_)
    # max is nbins-1
    bpos[bpos >= nbins] = 0

    return coef, bpos
    
def blinterp(img_size, cell_size):
    """
    Parameters : 
	------------
        img_size     : (int,int) - image shape
        cell_size    : (int,int) - cell size to determine ranges
        
    Return :
    --------
        coefficient and cell position surrounding
        
        where : 1 = coef_00 + coef_01 + coef_10 + coef_11
    """
    
    csize_y, csize_x = cell_size
    h, w = img_size
    h_incell, w_incell = h//csize_y, w//csize_x
    
    # build coefficients surrounding cells
    xi, yj = np.meshgrid(np.arange(w),np.arange(h))
   
    x_offset = (xi+(1+csize_x)//2)
    y_offset = (yj+(1+csize_y)//2)
    
    xi_i = (x_offset//csize_x)*csize_x
    yj_j = (y_offset//csize_y)*csize_y
    x_coef = (x_offset-xi_i)/csize_x
    y_coef = (y_offset-yj_j)/csize_y
    
    blin_coefs = np.asarray([
        (1-x_coef)*(1-y_coef),  \
        x_coef*(1-y_coef),      \
        (1-x_coef)*y_coef,      \
        x_coef*y_coef           
    ])

    # build cell mapper
    cell_mapper = xi//csize_x + (yj//csize_y) * w_incell
    cell_mapper_around = np.pad(cell_mapper, ((csize_y//2, csize_y//2),(csize_x//2, csize_x//2)), 'edge')
    
    cmapper = np.asarray([
        cell_mapper_around[:h,:w], \
        cell_mapper_around[:h,-w:], \
        cell_mapper_around[-h:,:w], \
        cell_mapper_around[-h:,-w:]
    ])
    
    return blin_coefs, cmapper
