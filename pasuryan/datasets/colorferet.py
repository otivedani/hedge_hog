from pathlib import Path
import re
import pandas as pd
import imageio
import numpy as np

def load_data(  base_dir="./dataset/colorferet",
                subset='smaller',
                poses=['fa', 'fb'],
                use_flags=True,
                label='id'
    ):
    """
    Parameters : 
    ------------
        base_dir:   string
                    where our colorferet data stored.
                    in this development, dvd1/ and dvd2/ dirs are merged under colorferet/ dir.
                    merged = path_contain_colorferet/colorferet/
                    not merged = path_contain_colorferet/colorferet/dvd1/ (and dvd2/), call this twice.
        subset: ['images','smaller','thumbnails']
                which image will be using. 'images' are the best resolution
        poses: ['fa','fb','pl','hl','ql','pr','hr','qr','ra','rb','rc','rd','re', [...]]
                (default)['fa','fb'] using front only
        use_flags:  boolean
                    using glass is OK? then True
        label:  [(default)'id', 'gender', 'race', 'dob']
                class target of sbject properties.
    see documentation for detail

    Return :
    --------
        img_data: array(subject_count, width, height, channels)
        img_target: array(subject_count, )

    """

    # param checking
    _pose_list = ['fa','fb','pl','hl','ql','pr','hr','qr','ra','rb','rc','rd','re']

    if subset not in ['images','smaller','thumbnails']:
        raise ValueError("must be one of : 'images','smaller','thumbnails'")

    if isinstance(poses, str) and poses in _pose_list:
        poses = [poses]
    elif isinstance(poses, list) and all(ps in _pose_list for ps in poses):
        pass
    else:
        raise ValueError("poses unknown")

    flag_ph = '()'
    if use_flags:
        flag_ph = '(_[abc])?'

    if label not in ['id', 'gender', 'race', 'yob']:
        raise ValueError("one of the : 'gender', 'race', 'yob' plz")

    colorferet_path = Path(base_dir)
    data_dir_path = colorferet_path.joinpath("data",subset)
    gtruth_dir_path = colorferet_path.joinpath("data","ground_truths","name_value")

    regx_dt = re.compile(r'.*(\d{5})[\\\/](\d{5})_(\d{6})_('+'|'.join(poses)+')'+flag_ph+'.ppm')
    
    # create list of filtered data
    ppl_data = []
    for dt_id_dir in data_dir_path.iterdir():
        for found_data in dt_id_dir.iterdir():
            try:
                subj_id, date, pose, flag = regx_dt.match(found_data.as_posix()).group(2,3,4,5)
                ppl_data += [(subj_id, date, pose, flag, found_data)]
            except:
                pass
    ppl_data.sort(key=lambda x: x[0]+x[1]+x[2])

    # create dictionary with id as key and value as properties
    desc = {}
    for gtru_id_dir in gtruth_dir_path.iterdir():
        with open(gtruth_dir_path/"{0}/{0}.txt".format(gtru_id_dir.stem), 'r') as fin:
            # desc += [{nameval.rstrip("\n").split("=")[0]:nameval.rstrip("\n").split("=")[1] for nameval in fin.readlines()}]
            sid = ''
            temp = {}
            for nameval in fin.readlines():
                nyaa = nameval.rstrip("\n").split("=")
                if nyaa[0] == 'id':
                    sid = nyaa[1].lstrip('cfrS')
                else:
                    temp[nyaa[0]] = nyaa[1]
            desc[sid] = temp
        
    # wrap it up
    _genders = { 'Male':1 , 'Female':0 }
    _races = {  'Asian':1 , 
                'Asian-Middle-Eastern':2 , 
                'Asian-Southern':3 , 
                'Black-or-African-American':4 , 
                'Hispanic':5 , 
                'White':6 , 
                'Native-American':7 , 
                'Pacific-Islander':8 , 
                'Other':0 }

    img_target = []
    img_data = []
    _target = None
    for subj_id,_,_,_,data_path in ppl_data:
        if label == 'gender':
            _target = _genders[desc[subj_id]['gender']]
        elif label == 'race':
            _target = _races[desc[subj_id]['gender']]
        elif label == 'dob':
            _target = desc[subj_id]['dob']
        else:
            _target = subj_id

        img_target.append(_target)
        img_data.append(imageio.imread(data_path))
    
    return np.array(img_data), np.array(img_target)
