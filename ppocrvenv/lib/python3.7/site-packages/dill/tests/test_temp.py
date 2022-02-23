#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
WINDOWS = sys.platform[:3] == 'win'


f = lambda x: x**2
x = [1,2,3,4,5]

# source code to tempfile
def test_code_to_tempfile():
    if not WINDOWS:  #see: https://bugs.python.org/issue14243
        pyfile = dump_source(f, alias='_f')
        _f = load_source(pyfile)
        assert _f(4) == f(4)

# source code to stream
def test_code_to_stream():
    pyfile = dumpIO_source(f, alias='_f')
    _f = loadIO_source(pyfile)
    assert _f(4) == f(4)

# pickle to tempfile
def test_pickle_to_tempfile():
    if not WINDOWS:  #see: https://bugs.python.org/issue14243
        dumpfile = dump(x)
        _x = load(dumpfile)
        assert _x == x

# pickle to stream
def test_pickle_to_stream():
    dumpfile = dumpIO(x)
    _x = loadIO(dumpfile)
    assert _x == x

### now testing the objects ###
f = lambda x: x**2
def g(x): return f(x) - x

def h(x):
  def g(x): return x
  return g(x) - x 

class Foo(object):
  def bar(self, x):
    return x*x+x
_foo = Foo()

def add(x,y):
  return x+y

# yes, same as 'f', but things are tricky when it comes to pointers
squared = lambda x:x**2

class Bar:
  pass
_bar = Bar()


# test function-type objects that take 2 args
def test_two_arg_functions():
  for obj in [add]:
    pyfile = dumpIO_source(obj, alias='_obj')
    _obj = loadIO_source(pyfile)
    assert _obj(4,2) == obj(4,2)

# test function-type objects that take 1 arg
def test_one_arg_functions():
  for obj in [g, h, squared]:
    pyfile = dumpIO_source(obj, alias='_obj')
    _obj = loadIO_source(pyfile)
    assert _obj(4) == obj(4)

# test instance-type objects
#for obj in [_bar, _foo]:
#  pyfile = dumpIO_source(obj, alias='_obj')
#  _obj = loadIO_source(pyfile)
#  assert type(_obj) == type(obj)

# test the rest of the objects
def test_the_rest():
  for obj in [Bar, Foo, Foo.bar, _foo.bar]:
    pyfile = dumpIO_source(obj, alias='_obj')
    _obj = loadIO_source(pyfile)
    assert _obj.__name__ == obj.__name__


if __name__ == '__main__':
    test_code_to_tempfile()
    test_code_to_stream()
    test_pickle_to_tempfile()
    test_pickle_to_stream()
    test_two_arg_functions()
    test_one_arg_functions()
    test_the_rest()
