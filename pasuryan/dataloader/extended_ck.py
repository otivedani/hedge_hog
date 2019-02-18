# dependencies
from pathlib import Path
import re

def extended_ck_load(xck_dir, side_dish='image'):
    """
    args
        xck_dir: extended cohn-kanade base dir, that contains either : cohn-kanade-images | Emotion | FACS
        side_dish: 
            'images' : (default) get all images only
            'emotion' : get images which have emotion label
            'facs' : get images which have facs data
        return array of dict(image,data) 

    structure of 'extended-cohn-kanade-images' is
        ./<type dir>/<key dir>/<index dir>/<the file>
        type dir : cohn-kanade-images | Emotion | FACS
        key : S\d{3}
        index : \d{3}
        the file : key_index_(\d+)_\.(png|txt)
    """
    # param adjust
    if side_dish is (not 'emotion' or not 'facs'):
        side_dish = 'image'
    
    _defdict = {
        'image':    {'prepath': 'cohn-kanade-images',   'suffext': '.png'},
        'emotion':  {'prepath': 'Emotion',              'suffext': '_emotion.txt'},
        'facs':     {'prepath': 'FACS',                 'suffext': '_facs.txt'}
    }
    
    basepath = Path(xck_dir)
    _crawlerpath = basepath /_defdict[side_dish]['prepath']
    
    _rgxext = re.compile(r"(.*)"+_defdict[side_dish]['suffext'])
    _rgxkid = re.compile(r"S(\d+)")

    ckdatas = []    
    imagepath = ''
    
    for _key in _crawlerpath.iterdir():
        if _key.is_dir() and _rgxkid.match(_key.name):
            # dir named key get!
            for _index in _key.iterdir():
                if _index.is_dir():
                    # dir named index get!
                    for _filepath in _index.iterdir():
                        # if _filepath.is_file() and re.match("(.*)"+_defdict[side_dish]['suffext'], _filepath.name):
                        if _filepath.is_file() and _rgxext.match(_filepath.name):
                            # file with proper extension get!
                            if side_dish is not 'image':
                                _strlabel = open(_filepath.as_posix(),"r").read()
                                # data = int(re.search('\s+(\d+)',_strlabel).group(1))
                                rawdata = str.split(_strlabel)
                                
                                _impath = _rgxext.sub(_defdict['image']['suffext'],_filepath.name)
                                imagepath = basepath/_defdict['image']['prepath']/_key.name/_index.name/_impath
                            else:
                                imagepath = _filepath
                                rawdata = ''
                            subject_id = int(_rgxkid.search(_key.name).group(1))
                            ckdatas.append({'imagepath':imagepath,'subject_id':subject_id,'rawdata':rawdata})
                            # print({'imagepath':imagepath,'rawdata':rawdata})
    
    return ckdatas

# res = extended_ck_load('./dataset/ck+/extended-cohn-kanade-images', 'image')
# # extended_ck_load('./dataset/ck+/extended-cohn-kanade-images', 'emotion')
# # res = extended_ck_load('./dataset/ck+/extended-cohn-kanade-images', 'facs')
# print len(res)
# for i in range(9):
#     print(res[i])


