#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import sys
import dill
import test_mixins as module
try: from imp import reload
except ImportError: pass
dill.settings['recurse'] = True

cached = (module.__cached__ if hasattr(module, "__cached__")
          else module.__file__.split(".", 1)[0] + ".pyc")

module.a = 1234

pik_mod = dill.dumps(module)

module.a = 0

# remove module
del sys.modules[module.__name__]
del module

module = dill.loads(pik_mod)
def test_attributes():
   #assert hasattr(module, "a") and module.a == 1234  #FIXME: -m dill.tests
    assert module.double_add(1, 2, 3) == 2 * module.fx

# Restart, and test use_diff

reload(module)

try:
    dill.use_diff()

    module.a = 1234

    pik_mod = dill.dumps(module)

    module.a = 0

    # remove module
    del sys.modules[module.__name__]
    del module

    module = dill.loads(pik_mod)
    def test_diff_attributes():
        assert hasattr(module, "a") and module.a == 1234
        assert module.double_add(1, 2, 3) == 2 * module.fx

except AttributeError:
    def test_diff_attributes():
        pass

# clean up
import os
if os.path.exists(cached):
    os.remove(cached)
pycache = os.path.join(os.path.dirname(module.__file__), "__pycache__")
if os.path.exists(pycache) and not os.listdir(pycache):
    os.removedirs(pycache)


# test when module is None
import math

def get_lambda(str, **kwarg):
    return eval(str, kwarg, None)

obj = get_lambda('lambda x: math.exp(x)', math=math)

def test_module_is_none():
    assert obj.__module__ is None
    assert dill.copy(obj)(3) == obj(3)


if __name__ == '__main__':
    test_attributes()
    test_diff_attributes()
    test_module_is_none()
