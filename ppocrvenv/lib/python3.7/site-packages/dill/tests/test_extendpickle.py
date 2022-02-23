#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill as pickle
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


def my_fn(x):
    return x * 17


def test_extend():
    obj = lambda : my_fn(34)
    assert obj() == 578

    obj_io = StringIO()
    pickler = pickle.Pickler(obj_io)
    pickler.dump(obj)

    obj_str = obj_io.getvalue()

    obj2_io = StringIO(obj_str)
    unpickler = pickle.Unpickler(obj2_io)
    obj2 = unpickler.load()

    assert obj2() == 578


def test_isdill():
    obj_io = StringIO()
    pickler = pickle.Pickler(obj_io)
    assert pickle._dill.is_dill(pickler) is True

    pickler = pickle._dill.StockPickler(obj_io)
    assert pickle._dill.is_dill(pickler) is False

    try:
        import multiprocess as mp
        pickler = mp.reduction.ForkingPickler(obj_io)
        assert pickle._dill.is_dill(pickler, child=True) is True
        assert pickle._dill.is_dill(pickler, child=False) is False
    except:
        pass


if __name__ == '__main__':
    test_extend()
    test_isdill()
