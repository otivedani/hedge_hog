# import skimage
import cv2

import yaml
import re
import os, shutil

#defaults
_HOGparam = {
    #universal params
    'wsz': (128, 128),
    'ppc' : (8, 8),
    'cpb' : (1, 1),
    'bstride' : (1, 1),
    'nbins' : 9,
    'signed' : False,
    #cv2 spesifics
    'derivAperture' : 1,
    'winSigma' : 4.,
    # 'histogramNormType' : 0,
    'L2HysThreshold' : 2.0000000000000001e-01,
    'gammaCorrection' : 0,
    'nlevels' : 64,
    #sk spesifics
    'vis': False,
    'norm': 'L2',
    #my spesifics
    'interp':True

}

def init_hogparam(HOGparamdict):
    global _HOGparam
    # print((cv2paramdict.iterkeys()))
    for key in HOGparamdict.keys():
        _HOGparam[key] = HOGparamdict[key]
    # print(cv2param)
    return

def get_cv_hogparam(path_to_file):
    global _HOGparam

    cv2hog = cv2.HOGDescriptor()
    cv2hog.save(path_to_file)
    shutil.copyfile(path_to_file,path_to_file+".old")
        
    cv2param = {
        'winSize' : [int(w) for w in iter(_HOGparam['wsz'])],
        'cellSize' : [int(c) for c in reversed(_HOGparam['ppc'])],
        'blockSize' : [int(c*b) for c,b in zip(_HOGparam['ppc'],_HOGparam['cpb'])],
        'blockStride' : [int(c*b) for c,b in zip(_HOGparam['ppc'],_HOGparam['bstride'])],
        'nbins' : _HOGparam['nbins'],        
        'signedGradient' : 0 if _HOGparam['signed'] else 1,
        'derivAperture' : _HOGparam['derivAperture'],
        'winSigma' : _HOGparam['winSigma'],
        'histogramNormType' : 0 if _HOGparam['norm']=='L2' else 0, #TODO change
        'L2HysThreshold' : _HOGparam['L2HysThreshold'],
        'gammaCorrection' : _HOGparam['gammaCorrection'],
        'nlevels' : _HOGparam['nlevels'],
    }
    
    with open(path_to_file+".old",'r') as infile:
        rxco = re.compile(r"^\s+(.*)\n")
        rxtab = re.compile(r"(.*)")
        rawstr = infile.readlines()
        headstr = ""
        for pstr in rawstr:
            if not rxco.match(pstr):
                # print(pstr)
               headstr += pstr 

        yamlstr = yaml.dump(cv2param)
        newstr = headstr + rxtab.sub(r"   \1",yamlstr)
        
    with open(path_to_file,'w') as outfile:
        # yaml.dump(newstr, outfile)
        outfile.truncate()
        outfile.write(newstr)
        # try:
        #     print(yaml.load(outfile))
        # except yaml.YAMLError as exc:
        #     print((exc))
    return
    

def get_sk_hogparam():
    global _HOGparam
    skHOGparam = {
                    'orientations':_HOGparam['nbins'],
                    'pixels_per_cell':_HOGparam['ppc'], 
                    'cells_per_block':_HOGparam['cpb'],
                    'block_norm':_HOGparam['norm'],
    }
    return skHOGparam

def get_my_hogparam():
    global _HOGparam
    myHOGparam = {
                    'cell_size':_HOGparam['ppc'],
                    'block_size':_HOGparam['cpb'],
                    'block_stride':_HOGparam['bstride'],
                    'nbins':_HOGparam['nbins'],
                    'useSigned':_HOGparam['signed'], 
                    'useInterpolation':_HOGparam['interp'],
    }
    return myHOGparam