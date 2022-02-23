#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
dill.settings['recurse'] = True

def f(func):
  def w(*args):
    return f(*args)
  return w

@f
def f2(): pass

# check when __main__ and on import
def test_decorated():
  assert dill.pickles(f2)


import doctest
import logging
logging.basicConfig(level=logging.DEBUG)

class SomeUnreferencedUnpicklableClass(object):
    def __reduce__(self):
        raise Exception

unpicklable = SomeUnreferencedUnpicklableClass()

# This works fine outside of Doctest:
def test_normal():
    serialized = dill.dumps(lambda x: x)

# should not try to pickle unpicklable object in __globals__
def tests():
    """
    >>> serialized = dill.dumps(lambda x: x)
    """
    return

#print("\n\nRunning Doctest:")
def test_doctest():
    doctest.testmod()


if __name__ == '__main__':
    test_decorated()
    test_normal()
    test_doctest()
