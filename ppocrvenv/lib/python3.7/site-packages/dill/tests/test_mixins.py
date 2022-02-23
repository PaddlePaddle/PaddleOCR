#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
dill.settings['recurse'] = True


def wtf(x,y,z):
  def zzz():
    return x
  def yyy():
    return y
  def xxx():
    return z
  return zzz,yyy


def quad(a=1, b=1, c=0):
  inverted = [False]
  def invert():
    inverted[0] = not inverted[0]
  def dec(f):
    def func(*args, **kwds):
      x = f(*args, **kwds)
      if inverted[0]: x = -x
      return a*x**2 + b*x + c
    func.__wrapped__ = f
    func.invert = invert
    func.inverted = inverted
    return func
  return dec


@quad(a=0,b=2)
def double_add(*args):
  return sum(args)


fx = sum([1,2,3])


### to make it interesting...
def quad_factory(a=1,b=1,c=0):
  def dec(f):
    def func(*args,**kwds):
      fx = f(*args,**kwds)
      return a*fx**2 + b*fx + c
    return func
  return dec


@quad_factory(a=0,b=4,c=0)
def quadish(x):
  return x+1


quadratic = quad_factory()


def doubler(f):
  def inner(*args, **kwds):
    fx = f(*args, **kwds)
    return 2*fx
  return inner


@doubler
def quadruple(x):
  return 2*x


def test_mixins():
  # test mixins
  assert double_add(1,2,3) == 2*fx
  double_add.invert()
  assert double_add(1,2,3) == -2*fx

  _d = dill.copy(double_add)
  assert _d(1,2,3) == -2*fx
 #_d.invert() #FIXME: fails seemingly randomly
 #assert _d(1,2,3) == 2*fx

  assert _d.__wrapped__(1,2,3) == fx

  # XXX: issue or feature? in python3.4, inverted is linked through copy
  if not double_add.inverted[0]:
      double_add.invert()

  # test some stuff from source and pointers
  ds = dill.source
  dd = dill.detect
  assert ds.getsource(dd.freevars(quadish)['f']) == '@quad_factory(a=0,b=4,c=0)\ndef quadish(x):\n  return x+1\n'
  assert ds.getsource(dd.freevars(quadruple)['f']) == '@doubler\ndef quadruple(x):\n  return 2*x\n'
  assert ds.importable(quadish, source=False) == 'from %s import quadish\n' % __name__
  assert ds.importable(quadruple, source=False) == 'from %s import quadruple\n' % __name__
  assert ds.importable(quadratic, source=False) == 'from %s import quadratic\n' % __name__
  assert ds.importable(double_add, source=False) == 'from %s import double_add\n' % __name__
  assert ds.importable(quadruple, source=True) == 'def doubler(f):\n  def inner(*args, **kwds):\n    fx = f(*args, **kwds)\n    return 2*fx\n  return inner\n\n@doubler\ndef quadruple(x):\n  return 2*x\n'
  #***** #FIXME: this needs work
  result = ds.importable(quadish, source=True)
  a,b,c,_,result = result.split('\n',4)
  assert result == 'def quad_factory(a=1,b=1,c=0):\n  def dec(f):\n    def func(*args,**kwds):\n      fx = f(*args,**kwds)\n      return a*fx**2 + b*fx + c\n    return func\n  return dec\n\n@quad_factory(a=0,b=4,c=0)\ndef quadish(x):\n  return x+1\n'
  assert set([a,b,c]) == set(['a = 0', 'c = 0', 'b = 4'])
  result = ds.importable(quadratic, source=True)
  a,b,c,result = result.split('\n',3)
  assert result == '\ndef dec(f):\n  def func(*args,**kwds):\n    fx = f(*args,**kwds)\n    return a*fx**2 + b*fx + c\n  return func\n'
  assert set([a,b,c]) == set(['a = 1', 'c = 0', 'b = 1'])
  result = ds.importable(double_add, source=True)
  a,b,c,d,_,result = result.split('\n',5)
  assert result == 'def quad(a=1, b=1, c=0):\n  inverted = [False]\n  def invert():\n    inverted[0] = not inverted[0]\n  def dec(f):\n    def func(*args, **kwds):\n      x = f(*args, **kwds)\n      if inverted[0]: x = -x\n      return a*x**2 + b*x + c\n    func.__wrapped__ = f\n    func.invert = invert\n    func.inverted = inverted\n    return func\n  return dec\n\n@quad(a=0,b=2)\ndef double_add(*args):\n  return sum(args)\n'
  assert set([a,b,c,d]) == set(['a = 0', 'c = 0', 'b = 2', 'inverted = [True]'])
  #*****


if __name__ == '__main__':
    test_mixins()
