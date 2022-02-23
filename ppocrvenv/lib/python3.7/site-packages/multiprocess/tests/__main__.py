#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/multiprocess/blob/master/LICENSE

from __future__ import print_function
import glob
import os
try:
    import pox
    python = pox.which_python(version=True, fullpath=False) or 'python'
except ImportError:
    python = 'python'
import subprocess as sp
from sys import platform
shell = platform[:3] == 'win'

suite = os.path.dirname(__file__) or os.path.curdir
tests = glob.glob(suite + os.path.sep + 'test_*.py')
tests = glob.glob(suite + os.path.sep + '__init__.py') + \
        [i for i in tests if 'main' not in i]


if __name__ == '__main__':

    for test in tests:
        p = sp.Popen([python, test], shell=shell).wait()
        if not p:
            print('.', end='')
    print('')

