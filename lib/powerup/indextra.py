#!/usr/bin/env python2
"""
in.dex : point out, show
dex.ter : right, skill
ex.ter : outer

in.d.ex.tra.py - numpy extension for 2darray loop vectorizer
"""
#Author @otivedani 

import numpy as np

def flatidx(*shapeargs):
    """
    construct ndarray of index, flat, C-ordering
    arow with ncol first (y,x)
    flatidx(...,[nrow,]ncol,)
    
    Parameters :
    ------------
    shapeargs : [...,[...,]] - shape of numpy ndarray

    
    Return : 
    --------
    arange() of numpy ndarray, shaped as origin

    """
    
    size = 1
    for sz in enumerate(shapeargs):
        size *= sz[1]
    return np.arange(size,dtype='int').reshape(shapeargs)

##========================================================
# divtocell
# target:: (4,6) -> (6,2,2) -> (3,2,2,2)
#TODO : fix below condition
# condition : image must be divideable by cell size, or else not gonna work.
def cellsidx(flatidx, pixels_per_cell, cells_per_block=None):
    """ 
    (lupa)
    
    Parameters :
    ------------
    
    Return : 
    --------
    
    
    """
    if(cells_per_block==None):
        cells_per_block = pixels_per_cell
    cell = flatidx[:pixels_per_cell[0],:pixels_per_cell[1]]
    # print cell.shape
    # # print "==============="
    # print np.add.outer(np.arange(0, image.shape[1], cell.shape[1]),cell)
    # # print "==============="
    # return np.add.outer(np.arange(0, image.ravel().shape[0], cell.shape[1]*image.shape[1]).T,\
    #     np.add.outer(np.arange(0, image.shape[1], cell.shape[1]),cell))
    cellblockpair = np.array([pixels_per_cell, cells_per_block])
    strideshape = np.int32(\
                np.floor((np.array(flatidx.shape)+(cellblockpair[1]-cellblockpair[0]))/cellblockpair[1])\
                )

    return np.add.outer(np.arange(0, strideshape[0]*cells_per_block[0]*flatidx.shape[1], cells_per_block[0]*flatidx.shape[1]).T,\
        np.add.outer(np.arange(0, strideshape[1]*cells_per_block[1], cells_per_block[1]),cell))

def convolver(image, cell_size, stride_size=None):
    """ 
    change view of numpy 2darray, manipulating strides
    
    Parameters :
    ------------
    image : numpy 2darray - image to be sliced
    cell_size : (int,int) - size of sub-image/cell to slice the image
    stride_size : (int,int) - how far must go to right+down 
                                if None (default), will be adjusted to cell_size
    
    Return : 
    --------
    view of numpy 2darray with shape (blocks, blocks, cell_h, cell_w)
    
    """
    if (stride_size == None):
        stride_size = cell_size

    strideshape = np.int32(\
                np.floor((np.array(image.shape)+(np.array(stride_size)-np.array(cell_size)))/np.array(stride_size))\
                )

    blkcol_stride = image.itemsize
    blkrow_stride = image.itemsize*stride_size[1]
    imgblkscol_stride = image.itemsize*image.shape[1]
    imgblksrow_stride = image.itemsize*stride_size[0]*image.shape[1]

    #sanitycheck
    # if(imgcellsrow_stride*imgcells_row/image.itemsize == image.shape[0]*image.shape[1]):
    #     print "pass"

    imgblocks = np.lib.stride_tricks.as_strided(image, \
                shape=(strideshape[0], strideshape[1], cell_size[0], cell_size[1]), \
                strides=(imgblksrow_stride,blkrow_stride,imgblkscol_stride,blkcol_stride) \
                )

    return imgblocks

def rowcolidx(row, col):
    """ 
    numpyzed 2d-loop
    
    Parameters :
    ------------
    row : int - row length
    col : int - col length
    
    Return : 
    --------
    r_ix, c_ix
    #(equal as)
    for r_ix in range(row):
        for c_ix in range(col):
            #do something with r_ix and c_ix
    
    """
    return np.repeat(np.arange(row, dtype='int'), col), np.tile(np.arange(col, dtype='int'), row)
