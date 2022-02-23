#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
demonstrate dill's ability to pickle different python types
test pickling of all Python Standard Library objects (currently: CH 1-14 @ 2.7)
"""

import dill as pickle
pickle.settings['recurse'] = True
#pickle.detect.trace(True)
#import pickle

# get all objects for testing
from dill import load_types, objects, extend
load_types(pickleable=True,unpickleable=False)

# uncomment the next two lines to test cloudpickle
#extend(False)
#import cloudpickle as pickle

# helper objects
class _class:
    def _method(self):
        pass

# objects that *fail* if imported
special = {}
special['LambdaType'] = _lambda = lambda x: lambda y: x
special['MethodType'] = _method = _class()._method
special['UnboundMethodType'] = _class._method
objects.update(special)

def pickles(name, exact=False):
    """quick check if object pickles with dill"""
    obj = objects[name]
    try:
        pik = pickle.loads(pickle.dumps(obj))
        if exact:
            try:
                assert pik == obj
            except AssertionError:
                assert type(obj) == type(pik)
                print ("weak: %s %s" % (name, type(obj)))
        else:
            assert type(obj) == type(pik)
    except Exception:
        print ("fails: %s %s" % (name, type(obj)))


def test_objects():
    for member in objects.keys():
       #pickles(member, exact=True)
        pickles(member, exact=False)


if __name__ == '__main__':
    test_objects()
