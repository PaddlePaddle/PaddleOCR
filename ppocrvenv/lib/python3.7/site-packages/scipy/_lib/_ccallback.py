from . import _ccallback_c

import ctypes

PyCFuncPtr = ctypes.CFUNCTYPE(ctypes.c_void_p).__bases__[0]

ffi = None

class CData:
    pass

def _import_cffi():
    global ffi, CData

    if ffi is not None:
        return

    try:
        import cffi
        ffi = cffi.FFI()
        CData = ffi.CData
    except ImportError:
        ffi = False


class LowLevelCallable(tuple):
    """
    Low-level callback function.

    Parameters
    ----------
    function : {PyCapsule, ctypes function pointer, cffi function pointer}
        Low-level callback function.
    user_data : {PyCapsule, ctypes void pointer, cffi void pointer}
        User data to pass on to the callback function.
    signature : str, optional
        Signature of the function. If omitted, determined from *function*,
        if possible.

    Attributes
    ----------
    function
        Callback function given.
    user_data
        User data given.
    signature
        Signature of the function.

    Methods
    -------
    from_cython
        Class method for constructing callables from Cython C-exported
        functions.

    Notes
    -----
    The argument ``function`` can be one of:

    - PyCapsule, whose name contains the C function signature
    - ctypes function pointer
    - cffi function pointer

    The signature of the low-level callback must match one of those expected
    by the routine it is passed to.

    If constructing low-level functions from a PyCapsule, the name of the
    capsule must be the corresponding signature, in the format::

        return_type (arg1_type, arg2_type, ...)

    For example::

        "void (double)"
        "double (double, int *, void *)"

    The context of a PyCapsule passed in as ``function`` is used as ``user_data``,
    if an explicit value for ``user_data`` was not given.

    """

    # Make the class immutable
    __slots__ = ()

    def __new__(cls, function, user_data=None, signature=None):
        # We need to hold a reference to the function & user data,
        # to prevent them going out of scope
        item = cls._parse_callback(function, user_data, signature)
        return tuple.__new__(cls, (item, function, user_data))

    def __repr__(self):
        return "LowLevelCallable({!r}, {!r})".format(self.function, self.user_data)

    @property
    def function(self):
        return tuple.__getitem__(self, 1)

    @property
    def user_data(self):
        return tuple.__getitem__(self, 2)

    @property
    def signature(self):
        return _ccallback_c.get_capsule_signature(tuple.__getitem__(self, 0))

    def __getitem__(self, idx):
        raise ValueError()

    @classmethod
    def from_cython(cls, module, name, user_data=None, signature=None):
        """
        Create a low-level callback function from an exported Cython function.

        Parameters
        ----------
        module : module
            Cython module where the exported function resides
        name : str
            Name of the exported function
        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
            User data to pass on to the callback function.
        signature : str, optional
            Signature of the function. If omitted, determined from *function*.

        """
        try:
            function = module.__pyx_capi__[name]
        except AttributeError as e:
            raise ValueError("Given module is not a Cython module with __pyx_capi__ attribute") from e
        except KeyError as e:
            raise ValueError("No function {!r} found in __pyx_capi__ of the module".format(name)) from e
        return cls(function, user_data, signature)

    @classmethod
    def _parse_callback(cls, obj, user_data=None, signature=None):
        _import_cffi()

        if isinstance(obj, LowLevelCallable):
            func = tuple.__getitem__(obj, 0)
        elif isinstance(obj, PyCFuncPtr):
            func, signature = _get_ctypes_func(obj, signature)
        elif isinstance(obj, CData):
            func, signature = _get_cffi_func(obj, signature)
        elif _ccallback_c.check_capsule(obj):
            func = obj
        else:
            raise ValueError("Given input is not a callable or a low-level callable (pycapsule/ctypes/cffi)")

        if isinstance(user_data, ctypes.c_void_p):
            context = _get_ctypes_data(user_data)
        elif isinstance(user_data, CData):
            context = _get_cffi_data(user_data)
        elif user_data is None:
            context = 0
        elif _ccallback_c.check_capsule(user_data):
            context = user_data
        else:
            raise ValueError("Given user data is not a valid low-level void* pointer (pycapsule/ctypes/cffi)")

        return _ccallback_c.get_raw_capsule(func, signature, context)


#
# ctypes helpers
#

def _get_ctypes_func(func, signature=None):
    # Get function pointer
    func_ptr = ctypes.cast(func, ctypes.c_void_p).value

    # Construct function signature
    if signature is None:
        signature = _typename_from_ctypes(func.restype) + " ("
        for j, arg in enumerate(func.argtypes):
            if j == 0:
                signature += _typename_from_ctypes(arg)
            else:
                signature += ", " + _typename_from_ctypes(arg)
        signature += ")"

    return func_ptr, signature


def _typename_from_ctypes(item):
    if item is None:
        return "void"
    elif item is ctypes.c_void_p:
        return "void *"

    name = item.__name__

    pointer_level = 0
    while name.startswith("LP_"):
        pointer_level += 1
        name = name[3:]

    if name.startswith('c_'):
        name = name[2:]

    if pointer_level > 0:
        name += " " + "*"*pointer_level

    return name


def _get_ctypes_data(data):
    # Get voidp pointer
    return ctypes.cast(data, ctypes.c_void_p).value


#
# CFFI helpers
#

def _get_cffi_func(func, signature=None):
    # Get function pointer
    func_ptr = ffi.cast('uintptr_t', func)

    # Get signature
    if signature is None:
        signature = ffi.getctype(ffi.typeof(func)).replace('(*)', ' ')

    return func_ptr, signature


def _get_cffi_data(data):
    # Get pointer
    return ffi.cast('uintptr_t', data)
