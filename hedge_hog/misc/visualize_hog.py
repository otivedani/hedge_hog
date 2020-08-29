"""
Helper to visualize HOG features
"""

import numpy as np
from ..feature import hog

from skimage import io, draw, feature
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt

def visualize_hog(hog_feature, img_shape, cell_size, nbins, vcell_sz=32):
    
    #height and weight in cells
    h_incell, w_incell =  (l//s for l,s in zip(img_shape, cell_size))

    #make lines according to angles
    lsbank = make_lines(nbins=nbins,frame_size=vcell_sz,degree_base=180)

    #make masks
    hog_masks = make_masks(hog_feature, vcell_sz)\
        *hog_feature[:, None, None]

    #apply masks to lines
    result = lsbank[None,:,:,:]*hog_masks.reshape(h_incell,w_incell,nbins,vcell_sz,vcell_sz)
    # print(result.shape)

    visualized = np.sum(result, axis=-3)#**(0.4545)
    visualized = np.clip(visualized, 0, 255)#**(0.4545)
    # print(np.amax(visualized))
    visualized = np.moveaxis(visualized,2,1).reshape(h_incell*vcell_sz,w_incell*vcell_sz)
    # print(visualized.shape)
    return visualized

def make_lines(
                nbins=9,
                degree_base=180,
                frame_size=8
                ):
    """
    #TODO docstring
    """

    # make nbins channel image
    lines_bank = np.zeros((nbins,frame_size,frame_size), dtype=np.uint8)
    # angles by bins, convert to radians
    theta_rad = (np.arange(nbins)*(degree_base/nbins))*(np.pi/180)
    # line diameter
    _d = frame_size-1
    # find m and c (y=mx+c) passing center point of visualization image (for now, using square (w=h))
    m = np.tan(theta_rad)
    c = (_d+1//2)*(1-m)

    # find max min y,x
    x_1 = np.abs(1+(_d)//2*(1+np.cos(theta_rad))).astype(np.int)
    y_1 = np.abs(1+(_d)//2*(1+np.sin(theta_rad))).astype(np.int)
    y_0 = _d - y_1
    x_0 = _d - x_1

    for i, img in enumerate(lines_bank):
        # using skimage
        # rr, cc = draw.line(x_1[i], y_0[i], x_0[i], y_1[i])
        # # x is y, y is -x (rotate by 90deg)
        # # rr, cc = draw.line(y_0[i], -x_0[i], y_1[i], -x_1[i])
        # img[rr,cc] = 255

        rr, cc, val = draw.line_aa(x_1[i], y_0[i], x_0[i], y_1[i])
        img[rr, cc] = val * 255

        # # using PIL
        # pilimg = Image.new('L', (frame_size, frame_size), 0)  
        # draw = ImageDraw.Draw(pilimg)
        # # draw.line([(x_1[i], y_0[i]), (x_0[i], y_1[i])], fill=255, width=2)
        # draw.line([(y_1[i], x_0[i]), (y_0[i], x_1[i])], fill=128, width=1)
        # # print(np.asarray(pilimg).shape,img.shape)
        # img += np.asarray(pilimg)

        #DEBUG
        # plt.imshow(img)
        # plt.show()
    
    return lines_bank


def make_masks(weights, rows, cols=None, shape='circle'):
    """
    Make masking image with shape and sizes

    #TODO docstring

    credit to : 
    https://github.com/javidcf

    original question : 
    https://stackoverflow.com/questions/54041467/

    """

    if cols is None:
        cols = rows
    # Add two dimensions to weights
    w = np.asarray(weights)[:, np.newaxis, np.newaxis]
    # Open grid of row and column coordinates
    r, c = np.ogrid[-1:1:rows * 1j, -1:1:cols * 1j]
    # Make masks where criteria is met
    if shape == 'square':
        return (np.abs(r) <= w) & (np.abs(c) <= w)
    else:
        # Arrays with distances to centre
        d = r * r + c * c
        return d <= w * w



