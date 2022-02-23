from __future__ import division, print_function, absolute_import

import doctest
import glob
import os
import unittest

try:
    import numpy as np
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass

pdir = os.path.pardir
docs_base = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            pdir, pdir, "doc", "source"))

files = glob.glob(os.path.join(docs_base, "*.rst")) + \
    glob.glob(os.path.join(docs_base, "*", "*.rst"))

suite = doctest.DocFileSuite(*files, module_relative=False, encoding="utf-8")


if __name__ == "__main__":
    unittest.TextTestRunner().run(suite)
