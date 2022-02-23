#
# Author: Pearu Peterson, March 2002
#

__all__ = ['get_flinalg_funcs']

# The following ensures that possibly missing flavor (C or Fortran) is
# replaced with the available one. If none is available, exception
# is raised at the first attempt to use the resources.
try:
    from . import _flinalg
except ImportError:
    _flinalg = None
#    from numpy.distutils.misc_util import PostponedException
#    _flinalg = PostponedException()
#    print _flinalg.__doc__
    has_column_major_storage = lambda a:0


def has_column_major_storage(arr):
    return arr.flags['FORTRAN']


_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}  # 'd' will be default for 'i',..


def get_flinalg_funcs(names,arrays=(),debug=0):
    """Return optimal available _flinalg function objects with
    names. Arrays are used to determine optimal prefix."""
    ordering = []
    for i in range(len(arrays)):
        t = arrays[i].dtype.char
        if t not in _type_conv:
            t = 'd'
        ordering.append((t,i))
    if ordering:
        ordering.sort()
        required_prefix = _type_conv[ordering[0][0]]
    else:
        required_prefix = 'd'
    # Some routines may require special treatment.
    # Handle them here before the default lookup.

    # Default lookup:
    if ordering and has_column_major_storage(arrays[ordering[0][1]]):
        suffix1,suffix2 = '_c','_r'
    else:
        suffix1,suffix2 = '_r','_c'

    funcs = []
    for name in names:
        func_name = required_prefix + name
        func = getattr(_flinalg,func_name+suffix1,
                       getattr(_flinalg,func_name+suffix2,None))
        funcs.append(func)
    return tuple(funcs)
