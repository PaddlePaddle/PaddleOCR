#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
testing some selected object types
"""

import dill
dill.settings['recurse'] = True

verbose = False

def test_dict_contents():
  c = type.__dict__
  for i,j in c.items():
   #try:
    ok = dill.pickles(j)
   #except:
   #  print ("FAIL: %s with %s" % (i, dill.detect.errors(j)))
    if verbose: print ("%s: %s, %s" % (ok, type(j), j))
    assert ok
  if verbose: print ("")

def _g(x): yield x;

def _f():
  try: raise
  except:
    from sys import exc_info
    e, er, tb = exc_info()
    return er, tb

class _d(object):
  def _method(self):
    pass

from dill import objects
from dill import load_types
load_types(pickleable=True,unpickleable=False)
_newclass = objects['ClassObjectType']
del objects

# getset_descriptor for new-style classes (fails on '_method', if not __main__)
def test_class_descriptors():
  d = _d.__dict__
  for i in d.values():
    ok = dill.pickles(i)
    if verbose: print ("%s: %s, %s" % (ok, type(i), i))
    assert ok
  if verbose: print ("")
  od = _newclass.__dict__
  for i in od.values():
    ok = dill.pickles(i)
    if verbose: print ("%s: %s, %s" % (ok, type(i), i))
    assert ok
  if verbose: print ("")

# (__main__) class instance for new-style classes
def test_class():
  o = _d()
  oo = _newclass()
  ok = dill.pickles(o)
  if verbose: print ("%s: %s, %s" % (ok, type(o), o))
  assert ok
  ok = dill.pickles(oo)
  if verbose: print ("%s: %s, %s" % (ok, type(oo), oo))
  assert ok
  if verbose: print ("")

# frames, generators, and tracebacks (all depend on frame)
def test_frame_related():
  g = _g(1)
  f = g.gi_frame
  e,t = _f()
  _is = lambda ok: not ok if dill._dill.IS_PYPY2 else ok
  ok = dill.pickles(f)
  if verbose: print ("%s: %s, %s" % (ok, type(f), f))
  assert _is(not ok) #XXX: dill fails
  ok = dill.pickles(g)
  if verbose: print ("%s: %s, %s" % (ok, type(g), g))
  assert _is(not ok) #XXX: dill fails
  ok = dill.pickles(t)
  if verbose: print ("%s: %s, %s" % (ok, type(t), t))
  assert not ok #XXX: dill fails
  ok = dill.pickles(e)
  if verbose: print ("%s: %s, %s" % (ok, type(e), e))
  assert ok
  if verbose: print ("")


if __name__ == '__main__':
  test_frame_related()
  test_dict_contents()
  test_class()
  test_class_descriptors()
