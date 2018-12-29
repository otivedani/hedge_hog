# dependencies
from pathlib import Path
import re

def yale_load(rootdir, selimdir, ispgm):
    """
    args : 
        rootdir : root dir of yale/
        selimdir : selected image dir 
            faces/ | yalefaces/ | etc.
        ispgm : load the .PGM file instead
    """
    rootPath = Path(rootdir)
    selim_Path = rootPath/selimdir
    patternstr = r"subject(\d{2})\.(\w+)"

    rgxpattern = re.compile(patternstr+"(\.pgm)" if ispgm else patternstr+"$")

    yaleresult = []

    for _file in selim_Path.iterdir():
        _result = rgxpattern.match(_file.name)
        if (_file.is_file() and _result):
            yaleresult.append({
            'imagepath':  _file,
            'subjectnumber': int(_result.group(1)),
            'data': _result.group(2)
            })

    return yaleresult

# rootdir = "./dataset/yale"
# selimdir = "faces"
# ispgm = True
# res = yale_load(rootdir, selimdir, ispgm)
# print res[0]
# print len(res)
