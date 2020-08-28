"""
examples on scikit-image : 

call : 
from skimage.feature import blob_dog, blob_log, blob_doh

structure :
skimage
    feature
        __init__.py (from .blob import blob_dog, blob_log, blob_doh)
        blob.py (contains blob_dog, blob_log, blob_doh)

conclusion : 
module imported because it was defined in module dir

"""


from .timemeter import timemeter