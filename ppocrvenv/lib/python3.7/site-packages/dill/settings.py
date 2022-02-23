#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
global settings for Pickler
"""

try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    from pickle import HIGHEST_PROTOCOL as DEFAULT_PROTOCOL

settings = {
   #'main' : None,
    'protocol' : DEFAULT_PROTOCOL,
    'byref' : False,
   #'strictio' : False,
    'fmode' : 0, #HANDLE_FMODE
    'recurse' : False,
    'ignore' : False,
}

del DEFAULT_PROTOCOL

