#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

from dill.detect import baditems, badobjects, badtypes, errors, parent, at, globalvars
from dill import settings
from dill._dill import IS_PYPY, IS_PYPY2
from pickle import PicklingError

import inspect
import sys

def test_bad_things():
    f = inspect.currentframe()
    assert baditems(f) == [f]
    #assert baditems(globals()) == [f] #XXX
    assert badobjects(f) is f
    assert badtypes(f) == type(f)
    assert type(errors(f)) is PicklingError if IS_PYPY2 else TypeError
    d = badtypes(f, 1)
    assert isinstance(d, dict)
    assert list(badobjects(f, 1).keys()) == list(d.keys())
    assert list(errors(f, 1).keys()) == list(d.keys())
    s = set([(err.__class__.__name__,err.args[0]) for err in list(errors(f, 1).values())])
    a = dict(s)
    if 0x30602f0 <= sys.hexversion > 0x30612f0: #XXX: travis-ci
        assert len(s) is len(a) # TypeError (and possibly PicklingError)
    n = 1 if IS_PYPY2 else 2
    assert len(a) is n if 'PicklingError' in a.keys() else n-1

def test_parent():
    x = [4,5,6,7]
    listiter = iter(x)
    obj = parent(listiter, list)
    assert obj is x

    if IS_PYPY: assert parent(obj, int) is None
    else: assert parent(obj, int) is x[-1] # python oddly? finds last int
    assert at(id(at)) is at

a, b, c = 1, 2, 3

def squared(x):
  return a+x**2

def foo(x):
  def bar(y):
    return squared(x)+y
  return bar

class _class:
    def _method(self):
        pass
    def ok(self):
        return True

def test_globals():
    def f():
        a
        def g():
            b
            def h():
                c
    assert globalvars(f) == dict(a=1, b=2, c=3)

    res = globalvars(foo, recurse=True)
    assert set(res) == set(['squared', 'a'])
    res = globalvars(foo, recurse=False)
    assert res == {}
    zap = foo(2)
    res = globalvars(zap, recurse=True)
    assert set(res) == set(['squared', 'a'])
    res = globalvars(zap, recurse=False)
    assert set(res) == set(['squared'])
    del zap
    res = globalvars(squared)
    assert set(res) == set(['a'])
    # FIXME: should find referenced __builtins__
    #res = globalvars(_class, recurse=True)
    #assert set(res) == set(['True'])
    #res = globalvars(_class, recurse=False)
    #assert res == {}
    #res = globalvars(_class.ok, recurse=True)
    #assert set(res) == set(['True'])
    #res = globalvars(_class.ok, recurse=False)
    #assert set(res) == set(['True'])


#98 dill ignores __getstate__ in interactive lambdas
bar = [0]

class Foo(object):
    def __init__(self):
        pass
    def __getstate__(self):
        bar[0] = bar[0]+1
        return {}
    def __setstate__(self, data):
        pass

f = Foo()

def test_getstate():
    from dill import dumps, loads
    dumps(f)
    b = bar[0]
    dumps(lambda: f, recurse=False) # doesn't call __getstate__
    assert bar[0] == b
    dumps(lambda: f, recurse=True) # calls __getstate__
    assert bar[0] == b + 1

#97 serialize lambdas in test files
def test_deleted():
    global sin
    from dill import dumps, loads
    from math import sin, pi

    def sinc(x):
        return sin(x)/x

    settings['recurse'] = True
    _sinc = dumps(sinc)
    sin = globals().pop('sin')
    sin = 1
    del sin
    sinc_ = loads(_sinc) # no NameError... pickling preserves 'sin'
    res = sinc_(1)
    from math import sin
    assert sinc(1) == res


def test_lambdify():
    try:
        from sympy import symbols, lambdify
    except ImportError:
        return
    settings['recurse'] = True
    x = symbols("x")
    y = x**2
    f = lambdify([x], y)
    z = min
    d = globals()
    globalvars(f, recurse=True, builtin=True)
    assert z is min 
    assert d is globals() 


if __name__ == '__main__':
    test_bad_things()
    test_parent()
    test_globals()
    test_getstate()
    test_deleted()
    test_lambdify()
