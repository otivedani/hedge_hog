import unittest
import numpy as np

import context
from modules.powerup import indextra

class FiltersTestCase(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(np.all(indextra.flatidx(9)-np.arange(9)),0)

    def test_rowcolidx(self):
        rix, cix = indextra.rowcolidx(3,2)
        wew = np.arange(6).reshape(3,2)
        self.assertEqual(np.all(wew[rix, cix] - wew.ravel()), 0)

if __name__ == '__main__':
    unittest.main()

