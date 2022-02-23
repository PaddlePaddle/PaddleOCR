#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
import sys
dill.settings['recurse'] = True


def is_py3():
    return hex(sys.hexversion) >= '0x30000f0'


def function_a(a):
    return a


def function_b(b, b1):
    return b + b1


def function_c(c, c1=1):
    return c + c1


def function_d(d, d1, d2=1):
    return d + d1 + d2


if is_py3():
    exec('''
def function_e(e, *e1, e2=1, e3=2):
    return e + sum(e1) + e2 + e3''')


def test_functions():
    dumped_func_a = dill.dumps(function_a)
    assert dill.loads(dumped_func_a)(0) == 0

    dumped_func_b = dill.dumps(function_b)
    assert dill.loads(dumped_func_b)(1,2) == 3

    dumped_func_c = dill.dumps(function_c)
    assert dill.loads(dumped_func_c)(1) == 2
    assert dill.loads(dumped_func_c)(1, 2) == 3

    dumped_func_d = dill.dumps(function_d)
    assert dill.loads(dumped_func_d)(1, 2) == 4
    assert dill.loads(dumped_func_d)(1, 2, 3) == 6
    assert dill.loads(dumped_func_d)(1, 2, d2=3) == 6

    if is_py3():
        exec('''
dumped_func_e = dill.dumps(function_e)
assert dill.loads(dumped_func_e)(1, 2) == 6
assert dill.loads(dumped_func_e)(1, 2, 3) == 9
assert dill.loads(dumped_func_e)(1, 2, e2=3) == 8
assert dill.loads(dumped_func_e)(1, 2, e2=3, e3=4) == 10
assert dill.loads(dumped_func_e)(1, 2, 3, e2=4) == 12
assert dill.loads(dumped_func_e)(1, 2, 3, e2=4, e3=5) == 15''')


if __name__ == '__main__':
    test_functions()
