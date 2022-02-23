#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
test dill's ability to handle nested functions
"""

import os
import math

import dill as pickle
pickle.settings['recurse'] = True


# the nested function: pickle should fail here, but dill is ok.
def adder(augend):
    zero = [0]

    def inner(addend):
        return addend + augend + zero[0]
    return inner


# rewrite the nested function using a class: standard pickle should work here.
class cadder(object):
    def __init__(self, augend):
        self.augend = augend
        self.zero = [0]

    def __call__(self, addend):
        return addend + self.augend + self.zero[0]


# rewrite again, but as an old-style class
class c2adder:
    def __init__(self, augend):
        self.augend = augend
        self.zero = [0]

    def __call__(self, addend):
        return addend + self.augend + self.zero[0]


# some basic class stuff
class basic(object):
    pass


class basic2:
    pass


x = 5
y = 1


def test_basic():
    a = [0, 1, 2]
    pa = pickle.dumps(a)
    pmath = pickle.dumps(math) #XXX: FAILS in pickle
    pmap = pickle.dumps(map)
    # ...
    la = pickle.loads(pa)
    lmath = pickle.loads(pmath)
    lmap = pickle.loads(pmap)
    assert list(map(math.sin, a)) == list(lmap(lmath.sin, la))


def test_basic_class():
    pbasic2 = pickle.dumps(basic2)
    _pbasic2 = pickle.loads(pbasic2)()
    pbasic = pickle.dumps(basic)
    _pbasic = pickle.loads(pbasic)()


def test_c2adder():
    pc2adder = pickle.dumps(c2adder)
    pc2add5 = pickle.loads(pc2adder)(x)
    assert pc2add5(y) == x+y


def test_pickled_cadder():
    pcadder = pickle.dumps(cadder)
    pcadd5 = pickle.loads(pcadder)(x)
    assert pcadd5(y) == x+y


def test_raw_adder_and_inner():
    add5 = adder(x)
    assert add5(y) == x+y


def test_pickled_adder():
    padder = pickle.dumps(adder)
    padd5 = pickle.loads(padder)(x)
    assert padd5(y) == x+y


def test_pickled_inner():
    add5 = adder(x)
    pinner = pickle.dumps(add5) #XXX: FAILS in pickle
    p5add = pickle.loads(pinner)
    assert p5add(y) == x+y


def test_moduledict_where_not_main():
    try:
        from . import test_moduledict
    except:
        import test_moduledict
    name = 'test_moduledict.py'
    if os.path.exists(name) and os.path.exists(name+'c'):
        os.remove(name+'c')

    if os.path.exists(name) and hasattr(test_moduledict, "__cached__") \
       and os.path.exists(test_moduledict.__cached__):
        os.remove(getattr(test_moduledict, "__cached__"))

    if os.path.exists("__pycache__") and not os.listdir("__pycache__"):
        os.removedirs("__pycache__")


if __name__ == '__main__':
    test_basic()
    test_basic_class()
    test_c2adder()
    test_pickled_cadder()
    test_raw_adder_and_inner()
    test_pickled_adder()
    test_pickled_inner()
    test_moduledict_where_not_main()
