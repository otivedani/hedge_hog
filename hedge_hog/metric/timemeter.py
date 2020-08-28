"""
TBA~
"""

import time
import gc

def timemeter(function, *args, **kwargs):
    #default value
    _kwargs = {'count': 1, 'comment': ''}
    _kwargs.update(kwargs)
    
    gc.disable()
    start = time.clock()
    for i in range(_kwargs['count']):
        result = function(*args)
    stop = time.clock()
    gc.enable()

    string = function.__name__ + ' time: ' + str(stop-start)

    if kwargs:
        string += " | on "
        for k in kwargs:
            string += "("+ k + "=" + str(_kwargs[k]) +")"

    print(string)

    return result

#TODO add : decorator "logtimeplz"
