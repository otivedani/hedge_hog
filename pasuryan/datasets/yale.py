# dependencies
from pathlib import Path
import re

def yale_image_load(thedir, dotpgm):
    """
    args : 
        rootdir : root dir of yale/
        selimdir : selected image dir 
            faces/ | yalefaces/ | etc.
        dotpgm : load the .PGM file
    """
    dirPath = Path(thedir)
    patternstr = r"subject(\d{2})\.(\w+)"

    rgxpattern = re.compile(patternstr+"(\.pgm)" if dotpgm else patternstr+"$")

    yaleresult = []

    for _file in dirPath.iterdir():
        _result = rgxpattern.match(_file.name)
        if (_file.is_file() and _result):
            yaleresult.append({
            'imagepath':  _file,
            'subject_id': int(_result.group(1)),
            'data': _result.group(2)
            })

    return yaleresult

# rootdir = "./dataset/yale"
# selimdir = "faces"
# ispgm = True
# res = yale_load(rootdir, selimdir, ispgm)
# print(res[0])
# print(len(res))
