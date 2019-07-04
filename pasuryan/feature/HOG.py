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
    nbins=9, useSigned=False, useInterpolation=True, normalizeType='L2', ravel=True):
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
    
    BLOCKNORM_F = {
        'L2': BlockNorm_L2,
        'L2-hys': BlockNorm_L2Hys,
        'L1': BlockNorm_L1,
        'L1-sqrt': BlockNorm_L1sqrt
    }
    if normalizeType not in BLOCKNORM_F:
        raise ValueError("Norm type not supported.")

    ## Section 0. Precompute variables
    degreebase = 180 if not useSigned else 360

    h_cell, w_cell = cell_size
    h, w = image2d.shape
    vert_cells, horz_cells = h//h_cell, w//w_cell
    
    # crop image
    if (h%h_cell != 0) or (w%w_cell != 0):
        h, w = vert_cells*h_cell, horz_cells*w_cell
        image2d = image2d[:h,:w]

    ## Section 1. gradient image x and y
    gX, gY = filters.simple_edge_gradients(image2d)
    
    ## Section 2. gradient magnitude and orientation
    mag, ori = filters.toPolar(gX, gY, signed=useSigned)

    ## Section 3. trilinear interpolation voting
    binStep = degreebase/nbins

    if useInterpolation:
        
        coef_2, bin_2 = linterp(ori, nbins, binStep)
        coef_4, cmap_4 = blinterp(image2d.shape, cell_size)
        
        magx = mag[None,None,:,:]*coef_2[:,None,:,:]*coef_4[None,:,:,:]
        binx = bin_2[:,None,:,:]+(cmap_4[None,:,:,:]*nbins)

    else:
        # build bin position, keep number between (0,nbins)
        bin_1 = ((ori+binStep/2)//binStep).astype(np.int_)
        bin_1[bin_1 >= nbins] = 0
        
        # build cell mapper
        xi, yj = np.meshgrid(np.arange(w),np.arange(h))
        cell_mapper = xi//w_cell + (yj//h_cell) * horz_cells
        cmap_1 = cell_mapper

        magx = mag[None,:,:]
        binx = bin_1[None,:,:]+(cmap_1[None,:,:]*nbins)
        
    hists = np.bincount(binx.ravel(), magx.ravel(), vert_cells*horz_cells*nbins)
    
    ## Section 4. block normalization
    alt_block_size = block_size[0], block_size[1]*nbins
    alt_block_stride = block_stride[0], block_stride[1]*nbins
    alt_blockhists = indextra.convolver(hists.reshape(vert_cells,-1), alt_block_size, alt_block_stride)
    blockhists = alt_blockhists.reshape(alt_blockhists.shape[0], alt_blockhists.shape[1], -1)
    
    out = BLOCKNORM_F[normalizeType](blockhists, eps=1e-7)
    
    ## Last : HOG feature
    if ravel:
        return out.ravel()
    else:
        return out

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
    c = (x-x_1)
    
    coef = np.asarray([c,(1-c)])
    bpos = np.asarray([x_1+1, x_1], dtype=np.int_)
    # keep number between (0,nbins)
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
    
    h_cell, w_cell = cell_size
    h, w = img_size
    vert_cells, horz_cells = h//h_cell, w//w_cell
    
    # build coefficients surrounding cells
    xi, yj = np.meshgrid(np.arange(w),np.arange(h))
   
    x_offset = (xi+(1+w_cell)//2)
    y_offset = (yj+(1+h_cell)//2)
    
    xi_i = (x_offset//w_cell)*w_cell
    yj_j = (y_offset//h_cell)*h_cell
    x_coef = (x_offset-xi_i)/w_cell
    y_coef = (y_offset-yj_j)/h_cell
    
    blin_coefs = np.asarray([
        (1-x_coef)*(1-y_coef),  \
        x_coef*(1-y_coef),      \
        (1-x_coef)*y_coef,      \
        x_coef*y_coef           
    ])

    # build cell mapper
    cell_mapper = xi//w_cell + (yj//h_cell) * horz_cells
    cell_mapper_around = np.pad(cell_mapper, ((h_cell//2, h_cell//2),(w_cell//2, w_cell//2)), 'edge')
    
    cmapper = np.asarray([
        cell_mapper_around[:h,:w],  \
        cell_mapper_around[:h,-w:], \
        cell_mapper_around[-h:,:w], \
        cell_mapper_around[-h:,-w:]
    ])
    
    return blin_coefs, cmapper

# Block Normalization Method
def BlockNorm_L2(blockhists, eps=1e-7):
    return blockhists / np.sqrt(np.sum(blockhists**2, axis=-1)+eps**2)[:,:,None]
def BlockNorm_L2Hys(blockhists, eps=1e-7, clip=0.2):
    out = BlockNorm_L2(blockhists, eps=eps)
    out[ out > clip ] = clip
    return BlockNorm_L2(out, eps=eps)
def BlockNorm_L1(blockhists, eps=1e-7):
    return blockhists / (np.sum(np.abs(blockhists), axis=-1)+eps)[:,:,None]
def BlockNorm_L1sqrt(blockhists, eps=1e-7):
    return np.sqrt(BlockNorm_L1(blockhists, eps=eps))
