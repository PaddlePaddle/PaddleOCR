#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
import sys
dill.settings['recurse'] = True

# test classdefs
class _class:
    def _method(self):
        pass
    def ok(self):
        return True

class _class2:
    def __call__(self):
        pass
    def ok(self):
        return True

class _newclass(object):
    def _method(self):
        pass
    def ok(self):
        return True

class _newclass2(object):
    def __call__(self):
        pass
    def ok(self):
        return True

class _meta(type):
    pass

def __call__(self):
    pass
def ok(self):
    return True

_mclass = _meta("_mclass", (object,), {"__call__": __call__, "ok": ok})

del __call__
del ok

o = _class()
oc = _class2()
n = _newclass()
nc = _newclass2()
m = _mclass()

# test pickles for class instances
def test_class_instances():
    assert dill.pickles(o)
    assert dill.pickles(oc)
    assert dill.pickles(n)
    assert dill.pickles(nc)
    assert dill.pickles(m)

def test_class_objects():
    clslist = [_class,_class2,_newclass,_newclass2,_mclass]
    objlist = [o,oc,n,nc,m]
    _clslist = [dill.dumps(obj) for obj in clslist]
    _objlist = [dill.dumps(obj) for obj in objlist]

    for obj in clslist:
        globals().pop(obj.__name__)
    del clslist
    for obj in ['o','oc','n','nc']:
        globals().pop(obj)
    del objlist
    del obj

    for obj,cls in zip(_objlist,_clslist):
        _cls = dill.loads(cls)
        _obj = dill.loads(obj)
        assert _obj.ok()
        assert _cls.ok(_cls())
        if _cls.__name__ == "_mclass":
            assert type(_cls).__name__ == "_meta"

# test NoneType
def test_none():
    assert dill.pickles(type(None))

if hex(sys.hexversion) >= '0x20600f0':
    from collections import namedtuple
    Z = namedtuple("Z", ['a','b'])
    Zi = Z(0,1)
    X = namedtuple("Y", ['a','b'])
    X.__name__ = "X"
    if hex(sys.hexversion) >= '0x30300f0':
        X.__qualname__ = "X" #XXX: name must 'match' or fails to pickle
    Xi = X(0,1)
    Bad = namedtuple("FakeName", ['a','b'])
    Badi = Bad(0,1)
else:
    Z = Zi = X = Xi = Bad = Badi = None

# test namedtuple
def test_namedtuple():
    assert Z is dill.loads(dill.dumps(Z))
    assert Zi == dill.loads(dill.dumps(Zi))
    assert X is dill.loads(dill.dumps(X))
    assert Xi == dill.loads(dill.dumps(Xi))
    assert Bad is not dill.loads(dill.dumps(Bad))
    assert Bad._fields == dill.loads(dill.dumps(Bad))._fields
    assert tuple(Badi) == tuple(dill.loads(dill.dumps(Badi)))

def test_array_nested():
    try:
        import numpy as np
    
        x = np.array([1])
        y = (x,)
        dill.dumps(x)
        assert y == dill.loads(dill.dumps(y))

    except ImportError: pass


def test_array_subclass():
    try:
        import numpy as np

        class TestArray(np.ndarray):
            def __new__(cls, input_array, color):
                obj = np.asarray(input_array).view(cls)
                obj.color = color
                return obj
            def __array_finalize__(self, obj):
                if obj is None:
                    return
                if isinstance(obj, type(self)):
                    self.color = obj.color
            def __getnewargs__(self):
                return np.asarray(self), self.color

        a1 = TestArray(np.zeros(100), color='green')
        if dill._dill.PY3 and not dill._dill.IS_PYPY:
            assert dill.pickles(a1)
            assert a1.__dict__ == dill.copy(a1).__dict__

        a2 = a1[0:9]
        if dill._dill.PY3 and not dill._dill.IS_PYPY:
            assert dill.pickles(a2)
            assert a2.__dict__ == dill.copy(a2).__dict__

        class TestArray2(np.ndarray):
            color = 'blue'

        a3 = TestArray2([1,2,3,4,5])
        a3.color = 'green'
        if dill._dill.PY3 and not dill._dill.IS_PYPY:
            assert dill.pickles(a3)
            assert a3.__dict__ == dill.copy(a3).__dict__

    except ImportError: pass


def test_method_decorator():
    class A(object):
      @classmethod
      def test(cls):
        pass

    a = A()

    res = dill.dumps(a)
    new_obj = dill.loads(res)
    new_obj.__class__.test()

# test slots
class Y(object):
  __slots__ = ['y']
  def __init__(self, y):
    self.y = y

value = 123
y = Y(value)

def test_slots():
    assert dill.pickles(Y)
    assert dill.pickles(y)
    assert dill.pickles(Y.y)
    assert dill.copy(y).y == value


if __name__ == '__main__':
    test_class_instances()
    test_class_objects()
    test_none()
    test_namedtuple()
    test_array_nested()
    test_array_subclass()
    test_method_decorator()
    test_slots()
