#!/usr/bin/env python
#
# Author: Kirill Makhonin (@kirillmakhonin)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill

class RestrictedType:
    def __bool__(*args, **kwargs):
        raise Exception('Restricted function')

    __eq__ = __lt__ = __le__ = __ne__ = __gt__ = __ge__ = __hash__ = __bool__

glob_obj = RestrictedType()

def restricted_func():
    a = glob_obj

def test_function_with_restricted_object():
    deserialized = dill.loads(dill.dumps(restricted_func, recurse=True))


if __name__ == '__main__':
    test_function_with_restricted_object()
