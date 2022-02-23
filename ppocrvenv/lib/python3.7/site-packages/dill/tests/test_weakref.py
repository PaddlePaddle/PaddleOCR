#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
dill.settings['recurse'] = True
import weakref

class _class:
    def _method(self):
        pass

class _class2:
    def __call__(self):
        pass

class _newclass(object):
    def _method(self):
        pass

class _newclass2(object):
    def __call__(self):
        pass

def _function():
    pass


def test_weakref():
    o = _class()
    oc = _class2()
    n = _newclass()
    nc = _newclass2()
    f = _function
    z = _class
    x = _newclass

    r = weakref.ref(o)
    dr = weakref.ref(_class())
    p = weakref.proxy(o)
    dp = weakref.proxy(_class())
    c = weakref.proxy(oc)
    dc = weakref.proxy(_class2())

    m = weakref.ref(n)
    dm = weakref.ref(_newclass())
    t = weakref.proxy(n)
    dt = weakref.proxy(_newclass())
    d = weakref.proxy(nc)
    dd = weakref.proxy(_newclass2())

    fr = weakref.ref(f)
    fp = weakref.proxy(f)
    #zr = weakref.ref(z) #XXX: weakrefs not allowed for classobj objects
    #zp = weakref.proxy(z) #XXX: weakrefs not allowed for classobj objects
    xr = weakref.ref(x)
    xp = weakref.proxy(x)

    objlist = [r,dr,m,dm,fr,xr, p,dp,t,dt, c,dc,d,dd, fp,xp]
    #dill.detect.trace(True)

    for obj in objlist:
      res = dill.detect.errors(obj)
      if res:
        print ("%s" % res)
       #print ("%s:\n  %s" % (obj, res))
    # else:
    #   print ("PASS: %s" % obj)
      assert not res

def test_dictproxy():
    from dill._dill import DictProxyType
    try:
        m = DictProxyType({"foo": "bar"})
    except:
        m = type.__dict__
    mp = dill.copy(m)   
    assert mp.items() == m.items()


if __name__ == '__main__':
    test_weakref()
    from dill._dill import IS_PYPY
    if not IS_PYPY:
        test_dictproxy()
