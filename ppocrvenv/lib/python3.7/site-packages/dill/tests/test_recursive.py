#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
from functools import partial
from dill._dill import PY3, OLDER
_super = super

class obj1(object):
    def __init__(self):
        super(obj1, self).__init__()

class obj2(object):
    def __init__(self):
        _super(obj2, self).__init__()

class obj3(object):
    super_ = super
    def __init__(self):
        obj3.super_(obj3, self).__init__()


def test_super():
    assert dill.copy(obj1(), byref=True)
    assert dill.copy(obj1(), byref=True, recurse=True)
   #assert dill.copy(obj1(), recurse=True) #FIXME: fails __main__.py
    assert dill.copy(obj1())

    assert dill.copy(obj2(), byref=True)
    assert dill.copy(obj2(), byref=True, recurse=True)
   #assert dill.copy(obj2(), recurse=True) #FIXME: fails __main__.py
    assert dill.copy(obj2())

    assert dill.copy(obj3(), byref=True)
    assert dill.copy(obj3(), byref=True, recurse=True)
   #assert dill.copy(obj3(), recurse=True) #FIXME: fails __main__.py
    assert dill.copy(obj3())


def get_trigger(model):
    pass

class Machine(object):
    def __init__(self):
        self.child = Model()
        self.trigger = partial(get_trigger, self)
        self.child.trigger = partial(get_trigger, self.child)

class Model(object):
    pass



def test_partial():
    assert dill.copy(Machine(), byref=True)
    assert dill.copy(Machine(), byref=True, recurse=True)
    if not OLDER:
        assert dill.copy(Machine(), recurse=True)
    assert dill.copy(Machine())


class Machine2(object):
    def __init__(self):
        self.go = partial(self.member, self)
    def member(self, model):
        pass


class SubMachine(Machine2):
    def __init__(self):
        _super(SubMachine, self).__init__()
        #super(SubMachine, self).__init__() #XXX: works, except for 3.1-3.3


def test_partials():
    assert dill.copy(SubMachine(), byref=True)
    assert dill.copy(SubMachine(), byref=True, recurse=True)
   #if not OLDER: #FIXME: fails __main__.py
   #    assert dill.copy(SubMachine(), recurse=True)
    assert dill.copy(SubMachine())



if __name__ == '__main__':
    #print(('byref','_super','_recurse','_memo','_stop','OLDER'))
    test_super()
    test_partial()
    test_partials()
