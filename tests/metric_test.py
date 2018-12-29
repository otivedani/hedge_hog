import unittest

import context
from modules.metric import timemeter

def fnwithargs(number):
    ret = 1 
    for i in range(number):
        ret *= 2**i
    return ret

def fntime():
    return fnwithargs(5)

class MetricTestCase(unittest.TestCase):
    def test_timemeter(self):
        timemeter(fntime, count=10000)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()



