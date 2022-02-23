import functools
import warnings

__all__ = ["_deprecated"]


def _deprecated(msg, stacklevel=2):
    """Deprecate a function by emitting a warning on use."""
    def wrap(fun):
        if isinstance(fun, type):
            warnings.warn(
                "Trying to deprecate class {!r}".format(fun),
                category=RuntimeWarning, stacklevel=2)
            return fun

        @functools.wraps(fun)
        def call(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning,
                          stacklevel=stacklevel)
            return fun(*args, **kwargs)
        call.__doc__ = msg
        return call

    return wrap


class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """
    def __init__(self, content, message):
        self._content = content
        self._message = message

    def __hash__(self):
        return hash(self._content)

    def __eq__(self, other):
        res = (self._content == other)
        if res:
            warnings.warn(self._message, category=DeprecationWarning,
                          stacklevel=2)
        return res


def deprecate_cython_api(module, routine_name, new_name=None, message=None):
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    old_name = "{}.{}".format(module.__name__, routine_name)

    if new_name is None:
        depdoc = "`%s` is deprecated!" % old_name
    else:
        depdoc = "`%s` is deprecated, use `%s` instead!" % \
                 (old_name, new_name)

    if message is not None:
        depdoc += "\n" + message

    d = module.__pyx_capi__

    # Check if the function is a fused-type function with a mangled name
    j = 0
    has_fused = False
    while True:
        fused_name = "__pyx_fuse_{}{}".format(j, routine_name)
        if fused_name in d:
            has_fused = True
            d[_DeprecationHelperStr(fused_name, depdoc)] = d.pop(fused_name)
            j += 1
        else:
            break

    # If not, apply deprecation to the named routine
    if not has_fused:
        d[_DeprecationHelperStr(routine_name, depdoc)] = d.pop(routine_name)

