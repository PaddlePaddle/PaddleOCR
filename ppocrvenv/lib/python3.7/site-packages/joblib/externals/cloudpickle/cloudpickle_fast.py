"""
New, fast version of the CloudPickler.

This new CloudPickler class can now extend the fast C Pickler instead of the
previous Python implementation of the Pickler class. Because this functionality
is only available for Python versions 3.8+, a lot of backward-compatibility
code is also removed.

Note that the C Pickler subclassing API is CPython-specific. Therefore, some
guards present in cloudpickle.py that were written to handle PyPy specificities
are not present in cloudpickle_fast.py
"""
import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing

from enum import Enum
from collections import ChainMap, OrderedDict

from .compat import pickle, Pickler
from .cloudpickle import (
    _extract_code_globals, _BUILTIN_TYPE_NAMES, DEFAULT_PROTOCOL,
    _find_imported_submodules, _get_cell_contents, _should_pickle_by_reference,
    _builtin_type, _get_or_create_tracker_id,  _make_skeleton_class,
    _make_skeleton_enum, _extract_class_dict, dynamic_subimport, subimport,
    _typevar_reduce, _get_bases, _make_cell, _make_empty_cell, CellType,
    _is_parametrized_type_hint, PYPY, cell_set,
    parametrized_type_hint_getinitargs, _create_parametrized_type_hint,
    builtin_code_type,
    _make_dict_keys, _make_dict_values, _make_dict_items,
)


if pickle.HIGHEST_PROTOCOL >= 5 and not PYPY:
    # Shorthands similar to pickle.dump/pickle.dumps

    def dump(obj, file, protocol=None, buffer_callback=None):
        """Serialize obj as bytes streamed into file

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        CloudPickler(
            file, protocol=protocol, buffer_callback=buffer_callback
        ).dump(obj)

    def dumps(obj, protocol=None, buffer_callback=None):
        """Serialize obj as a string of bytes allocated in memory

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        with io.BytesIO() as file:
            cp = CloudPickler(
                file, protocol=protocol, buffer_callback=buffer_callback
            )
            cp.dump(obj)
            return file.getvalue()

else:
    # Shorthands similar to pickle.dump/pickle.dumps
    def dump(obj, file, protocol=None):
        """Serialize obj as bytes streamed into file

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        CloudPickler(file, protocol=protocol).dump(obj)

    def dumps(obj, protocol=None):
        """Serialize obj as a string of bytes allocated in memory

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        with io.BytesIO() as file:
            cp = CloudPickler(file, protocol=protocol)
            cp.dump(obj)
            return file.getvalue()


load, loads = pickle.load, pickle.loads


# COLLECTION OF OBJECTS __getnewargs__-LIKE METHODS
# -------------------------------------------------

def _class_getnewargs(obj):
    type_kwargs = {}
    if "__slots__" in obj.__dict__:
        type_kwargs["__slots__"] = obj.__slots__

    __dict__ = obj.__dict__.get('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__

    return (type(obj), obj.__name__, _get_bases(obj), type_kwargs,
            _get_or_create_tracker_id(obj), None)


def _enum_getnewargs(obj):
    members = dict((e.name, e.value) for e in obj)
    return (obj.__bases__, obj.__name__, obj.__qualname__, members,
            obj.__module__, _get_or_create_tracker_id(obj), None)


# COLLECTION OF OBJECTS RECONSTRUCTORS
# ------------------------------------
def _file_reconstructor(retval):
    return retval


# COLLECTION OF OBJECTS STATE GETTERS
# -----------------------------------
def _function_getstate(func):
    # - Put func's dynamic attributes (stored in func.__dict__) in state. These
    #   attributes will be restored at unpickling time using
    #   f.__dict__.update(state)
    # - Put func's members into slotstate. Such attributes will be restored at
    #   unpickling time by iterating over slotstate and calling setattr(func,
    #   slotname, slotvalue)
    slotstate = {
        "__name__": func.__name__,
        "__qualname__": func.__qualname__,
        "__annotations__": func.__annotations__,
        "__kwdefaults__": func.__kwdefaults__,
        "__defaults__": func.__defaults__,
        "__module__": func.__module__,
        "__doc__": func.__doc__,
        "__closure__": func.__closure__,
    }

    f_globals_ref = _extract_code_globals(func.__code__)
    f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in
                 func.__globals__}

    closure_values = (
        list(map(_get_cell_contents, func.__closure__))
        if func.__closure__ is not None else ()
    )

    # Extract currently-imported submodules used by func. Storing these modules
    # in a smoke _cloudpickle_subimports attribute of the object's state will
    # trigger the side effect of importing these modules at unpickling time
    # (which is necessary for func to work correctly once depickled)
    slotstate["_cloudpickle_submodules"] = _find_imported_submodules(
        func.__code__, itertools.chain(f_globals.values(), closure_values))
    slotstate["__globals__"] = f_globals

    state = func.__dict__
    return state, slotstate


def _class_getstate(obj):
    clsdict = _extract_class_dict(obj)
    clsdict.pop('__weakref__', None)

    if issubclass(type(obj), abc.ABCMeta):
        # If obj is an instance of an ABCMeta subclass, don't pickle the
        # cache/negative caches populated during isinstance/issubclass
        # checks, but pickle the list of registered subclasses of obj.
        clsdict.pop('_abc_cache', None)
        clsdict.pop('_abc_negative_cache', None)
        clsdict.pop('_abc_negative_cache_version', None)
        registry = clsdict.pop('_abc_registry', None)
        if registry is None:
            # in Python3.7+, the abc caches and registered subclasses of a
            # class are bundled into the single _abc_impl attribute
            clsdict.pop('_abc_impl', None)
            (registry, _, _, _) = abc._get_dump(obj)

            clsdict["_abc_impl"] = [subclass_weakref()
                                    for subclass_weakref in registry]
        else:
            # In the above if clause, registry is a set of weakrefs -- in
            # this case, registry is a WeakSet
            clsdict["_abc_impl"] = [type_ for type_ in registry]

    if "__slots__" in clsdict:
        # pickle string length optimization: member descriptors of obj are
        # created automatically from obj's __slots__ attribute, no need to
        # save them in obj's state
        if isinstance(obj.__slots__, str):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)

    clsdict.pop('__dict__', None)  # unpicklable property object

    return (clsdict, {})


def _enum_getstate(obj):
    clsdict, slotstate = _class_getstate(obj)

    members = dict((e.name, e.value) for e in obj)
    # Cleanup the clsdict that will be passed to _rehydrate_skeleton_class:
    # Those attributes are already handled by the metaclass.
    for attrname in ["_generate_next_value_", "_member_names_",
                     "_member_map_", "_member_type_",
                     "_value2member_map_"]:
        clsdict.pop(attrname, None)
    for member in members:
        clsdict.pop(member)
        # Special handling of Enum subclasses
    return clsdict, slotstate


# COLLECTIONS OF OBJECTS REDUCERS
# -------------------------------
# A reducer is a function taking a single argument (obj), and that returns a
# tuple with all the necessary data to re-construct obj. Apart from a few
# exceptions (list, dict, bytes, int, etc.), a reducer is necessary to
# correctly pickle an object.
# While many built-in objects (Exceptions objects, instances of the "object"
# class, etc), are shipped with their own built-in reducer (invoked using
# obj.__reduce__), some do not. The following methods were created to "fill
# these holes".

def _code_reduce(obj):
    """codeobject reducer"""
    if hasattr(obj, "co_linetable"):  # pragma: no branch
        # Python 3.10 and later: obj.co_lnotab is deprecated and constructor
        # expects obj.co_linetable instead.
        args = (
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_linetable, obj.co_freevars,
            obj.co_cellvars
        )
    elif hasattr(obj, "co_posonlyargcount"):
        # Backward compat for 3.9 and older
        args = (
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
            obj.co_cellvars
        )
    else:
        # Backward compat for even older versions of Python
        args = (
            obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals,
            obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts,
            obj.co_names, obj.co_varnames, obj.co_filename,
            obj.co_name, obj.co_firstlineno, obj.co_lnotab,
            obj.co_freevars, obj.co_cellvars
        )
    return types.CodeType, args


def _cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer"""
    try:
        obj.cell_contents
    except ValueError:  # cell is empty
        return _make_empty_cell, ()
    else:
        return _make_cell, (obj.cell_contents, )


def _classmethod_reduce(obj):
    orig_func = obj.__func__
    return type(obj), (orig_func,)


def _file_reduce(obj):
    """Save a file"""
    import io

    if not hasattr(obj, "name") or not hasattr(obj, "mode"):
        raise pickle.PicklingError(
            "Cannot pickle files that do not map to an actual file"
        )
    if obj is sys.stdout:
        return getattr, (sys, "stdout")
    if obj is sys.stderr:
        return getattr, (sys, "stderr")
    if obj is sys.stdin:
        raise pickle.PicklingError("Cannot pickle standard input")
    if obj.closed:
        raise pickle.PicklingError("Cannot pickle closed files")
    if hasattr(obj, "isatty") and obj.isatty():
        raise pickle.PicklingError(
            "Cannot pickle files that map to tty objects"
        )
    if "r" not in obj.mode and "+" not in obj.mode:
        raise pickle.PicklingError(
            "Cannot pickle files that are not opened for reading: %s"
            % obj.mode
        )

    name = obj.name

    retval = io.StringIO()

    try:
        # Read the whole file
        curloc = obj.tell()
        obj.seek(0)
        contents = obj.read()
        obj.seek(curloc)
    except IOError as e:
        raise pickle.PicklingError(
            "Cannot pickle file %s as it cannot be read" % name
        ) from e
    retval.write(contents)
    retval.seek(curloc)

    retval.name = name
    return _file_reconstructor, (retval,)


def _getset_descriptor_reduce(obj):
    return getattr, (obj.__objclass__, obj.__name__)


def _mappingproxy_reduce(obj):
    return types.MappingProxyType, (dict(obj),)


def _memoryview_reduce(obj):
    return bytes, (obj.tobytes(),)


def _module_reduce(obj):
    if _should_pickle_by_reference(obj):
        return subimport, (obj.__name__,)
    else:
        # Some external libraries can populate the "__builtins__" entry of a
        # module's `__dict__` with unpicklable objects (see #316). For that
        # reason, we do not attempt to pickle the "__builtins__" entry, and
        # restore a default value for it at unpickling time.
        state = obj.__dict__.copy()
        state.pop('__builtins__', None)
        return dynamic_subimport, (obj.__name__, state)


def _method_reduce(obj):
    return (types.MethodType, (obj.__func__, obj.__self__))


def _logger_reduce(obj):
    return logging.getLogger, (obj.name,)


def _root_logger_reduce(obj):
    return logging.getLogger, ()


def _property_reduce(obj):
    return property, (obj.fget, obj.fset, obj.fdel, obj.__doc__)


def _weakset_reduce(obj):
    return weakref.WeakSet, (list(obj),)


def _dynamic_class_reduce(obj):
    """
    Save a class that can't be stored as module global.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from global modules.
    """
    if Enum is not None and issubclass(obj, Enum):
        return (
            _make_skeleton_enum, _enum_getnewargs(obj), _enum_getstate(obj),
            None, None, _class_setstate
        )
    else:
        return (
            _make_skeleton_class, _class_getnewargs(obj), _class_getstate(obj),
            None, None, _class_setstate
        )


def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    if obj is type(None):  # noqa
        return type, (None,)
    elif obj is type(Ellipsis):
        return type, (Ellipsis,)
    elif obj is type(NotImplemented):
        return type, (NotImplemented,)
    elif obj in _BUILTIN_TYPE_NAMES:
        return _builtin_type, (_BUILTIN_TYPE_NAMES[obj],)
    elif not _should_pickle_by_reference(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented


def _dict_keys_reduce(obj):
    # Safer not to ship the full dict as sending the rest might
    # be unintended and could potentially cause leaking of
    # sensitive information
    return _make_dict_keys, (list(obj), )


def _dict_values_reduce(obj):
    # Safer not to ship the full dict as sending the rest might
    # be unintended and could potentially cause leaking of
    # sensitive information
    return _make_dict_values, (list(obj), )


def _dict_items_reduce(obj):
    return _make_dict_items, (dict(obj), )


def _odict_keys_reduce(obj):
    # Safer not to ship the full dict as sending the rest might
    # be unintended and could potentially cause leaking of
    # sensitive information
    return _make_dict_keys, (list(obj), True)


def _odict_values_reduce(obj):
    # Safer not to ship the full dict as sending the rest might
    # be unintended and could potentially cause leaking of
    # sensitive information
    return _make_dict_values, (list(obj), True)


def _odict_items_reduce(obj):
    return _make_dict_items, (dict(obj), True)


# COLLECTIONS OF OBJECTS STATE SETTERS
# ------------------------------------
# state setters are called at unpickling time, once the object is created and
# it has to be updated to how it was at unpickling time.


def _function_setstate(obj, state):
    """Update the state of a dynamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
    state, slotstate = state
    obj.__dict__.update(state)

    obj_globals = slotstate.pop("__globals__")
    obj_closure = slotstate.pop("__closure__")
    # _cloudpickle_subimports is a set of submodules that must be loaded for
    # the pickled function to work correctly at unpickling time. Now that these
    # submodules are depickled (hence imported), they can be removed from the
    # object's state (the object state only served as a reference holder to
    # these submodules)
    slotstate.pop("_cloudpickle_submodules")

    obj.__globals__.update(obj_globals)
    obj.__globals__["__builtins__"] = __builtins__

    if obj_closure is not None:
        for i, cell in enumerate(obj_closure):
            try:
                value = cell.cell_contents
            except ValueError:  # cell is empty
                continue
            cell_set(obj.__closure__[i], value)

    for k, v in slotstate.items():
        setattr(obj, k, v)


def _class_setstate(obj, state):
    state, slotstate = state
    registry = None
    for attrname, attr in state.items():
        if attrname == "_abc_impl":
            registry = attr
        else:
            setattr(obj, attrname, attr)
    if registry is not None:
        for subclass in registry:
            obj.register(subclass)

    return obj


class CloudPickler(Pickler):
    # set of reducers defined and used by cloudpickle (private)
    _dispatch_table = {}
    _dispatch_table[classmethod] = _classmethod_reduce
    _dispatch_table[io.TextIOWrapper] = _file_reduce
    _dispatch_table[logging.Logger] = _logger_reduce
    _dispatch_table[logging.RootLogger] = _root_logger_reduce
    _dispatch_table[memoryview] = _memoryview_reduce
    _dispatch_table[property] = _property_reduce
    _dispatch_table[staticmethod] = _classmethod_reduce
    _dispatch_table[CellType] = _cell_reduce
    _dispatch_table[types.CodeType] = _code_reduce
    _dispatch_table[types.GetSetDescriptorType] = _getset_descriptor_reduce
    _dispatch_table[types.ModuleType] = _module_reduce
    _dispatch_table[types.MethodType] = _method_reduce
    _dispatch_table[types.MappingProxyType] = _mappingproxy_reduce
    _dispatch_table[weakref.WeakSet] = _weakset_reduce
    _dispatch_table[typing.TypeVar] = _typevar_reduce
    _dispatch_table[_collections_abc.dict_keys] = _dict_keys_reduce
    _dispatch_table[_collections_abc.dict_values] = _dict_values_reduce
    _dispatch_table[_collections_abc.dict_items] = _dict_items_reduce
    _dispatch_table[type(OrderedDict().keys())] = _odict_keys_reduce
    _dispatch_table[type(OrderedDict().values())] = _odict_values_reduce
    _dispatch_table[type(OrderedDict().items())] = _odict_items_reduce


    dispatch_table = ChainMap(_dispatch_table, copyreg.dispatch_table)

    # function reducers are defined as instance methods of CloudPickler
    # objects, as they rely on a CloudPickler attribute (globals_ref)
    def _dynamic_function_reduce(self, func):
        """Reduce a function that is not pickleable via attribute lookup."""
        newargs = self._function_getnewargs(func)
        state = _function_getstate(func)
        return (types.FunctionType, newargs, state, None, None,
                _function_setstate)

    def _function_reduce(self, obj):
        """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this
        reducer returns NotImplemented, making the CloudPickler fallback to
        traditional _pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.

        As opposed to cloudpickle.py, There no special handling for builtin
        pypy functions because cloudpickle_fast is CPython-specific.
        """
        if _should_pickle_by_reference(obj):
            return NotImplemented
        else:
            return self._dynamic_function_reduce(obj)

    def _function_getnewargs(self, func):
        code = func.__code__

        # base_globals represents the future global namespace of func at
        # unpickling time. Looking it up and storing it in
        # CloudpiPickler.globals_ref allow functions sharing the same globals
        # at pickling time to also share them once unpickled, at one condition:
        # since globals_ref is an attribute of a CloudPickler instance, and
        # that a new CloudPickler is created each time pickle.dump or
        # pickle.dumps is called, functions also need to be saved within the
        # same invocation of cloudpickle.dump/cloudpickle.dumps (for example:
        # cloudpickle.dumps([f1, f2])). There is no such limitation when using
        # CloudPickler.dump, as long as the multiple invocations are bound to
        # the same CloudPickler.
        base_globals = self.globals_ref.setdefault(id(func.__globals__), {})

        if base_globals == {}:
            # Add module attributes used to resolve relative imports
            # instructions inside func.
            for k in ["__package__", "__name__", "__path__", "__file__"]:
                if k in func.__globals__:
                    base_globals[k] = func.__globals__[k]

        # Do not bind the free variables before the function is created to
        # avoid infinite recursion.
        if func.__closure__ is None:
            closure = None
        else:
            closure = tuple(
                _make_empty_cell() for _ in range(len(code.co_freevars)))

        return code, base_globals, None, None, closure

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if "recursion" in e.args[0]:
                msg = (
                    "Could not pickle object as excessively deep recursion "
                    "required."
                )
                raise pickle.PicklingError(msg) from e
            else:
                raise

    if pickle.HIGHEST_PROTOCOL >= 5:
        # `CloudPickler.dispatch` is only left for backward compatibility - note
        # that when using protocol 5, `CloudPickler.dispatch` is not an
        # extension of `Pickler.dispatch` dictionary, because CloudPickler
        # subclasses the C-implemented Pickler, which does not expose a
        # `dispatch` attribute.  Earlier versions of the protocol 5 CloudPickler
        # used `CloudPickler.dispatch` as a class-level attribute storing all
        # reducers implemented by cloudpickle, but the attribute name was not a
        # great choice given the meaning of `CloudPickler.dispatch` when
        # `CloudPickler` extends the pure-python pickler.
        dispatch = dispatch_table

        # Implementation of the reducer_override callback, in order to
        # efficiently serialize dynamic functions and classes by subclassing
        # the C-implemented Pickler.
        # TODO: decorrelate reducer_override (which is tied to CPython's
        # implementation - would it make sense to backport it to pypy? - and
        # pickle's protocol 5 which is implementation agnostic. Currently, the
        # availability of both notions coincide on CPython's pickle and the
        # pickle5 backport, but it may not be the case anymore when pypy
        # implements protocol 5
        def __init__(self, file, protocol=None, buffer_callback=None):
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(
                self, file, protocol=protocol, buffer_callback=buffer_callback
            )
            # map functions __globals__ attribute ids, to ensure that functions
            # sharing the same global namespace at pickling time also share
            # their global namespace at unpickling time.
            self.globals_ref = {}
            self.proto = int(protocol)

        def reducer_override(self, obj):
            """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C _pickle.Pickler class
            cannot register custom reducers for functions and classes in the
            dispatch_table. Reducer for such types must instead implemented in
            the special reducer_override method.

            Note that method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.


            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):  # noqa  # pragma: no branch
                return (
                    _create_parametrized_type_hint,
                    parametrized_type_hint_getinitargs(obj)
                )
            t = type(obj)
            try:
                is_anyclass = issubclass(t, type)
            except TypeError:  # t is not a class (old Boost; see SF #502085)
                is_anyclass = False

            if is_anyclass:
                return _class_reduce(obj)
            elif isinstance(obj, types.FunctionType):
                return self._function_reduce(obj)
            else:
                # fallback to save_global, including the Pickler's
                # dispatch_table
                return NotImplemented

    else:
        # When reducer_override is not available, hack the pure-Python
        # Pickler's types.FunctionType and type savers. Note: the type saver
        # must override Pickler.save_global, because pickle.py contains a
        # hard-coded call to save_global when pickling meta-classes.
        dispatch = Pickler.dispatch.copy()

        def __init__(self, file, protocol=None):
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(self, file, protocol=protocol)
            # map functions __globals__ attribute ids, to ensure that functions
            # sharing the same global namespace at pickling time also share
            # their global namespace at unpickling time.
            self.globals_ref = {}
            assert hasattr(self, 'proto')

        def _save_reduce_pickle5(self, func, args, state=None, listitems=None,
                                 dictitems=None, state_setter=None, obj=None):
            save = self.save
            write = self.write
            self.save_reduce(
                func, args, state=None, listitems=listitems,
                dictitems=dictitems, obj=obj
            )
            # backport of the Python 3.8 state_setter pickle operations
            save(state_setter)
            save(obj)  # simple BINGET opcode as obj is already memoized.
            save(state)
            write(pickle.TUPLE2)
            # Trigger a state_setter(obj, state) function call.
            write(pickle.REDUCE)
            # The purpose of state_setter is to carry-out an
            # inplace modification of obj. We do not care about what the
            # method might return, so its output is eventually removed from
            # the stack.
            write(pickle.POP)

        def save_global(self, obj, name=None, pack=struct.pack):
            """
            Save a "global".

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
            if obj is type(None):  # noqa
                return self.save_reduce(type, (None,), obj=obj)
            elif obj is type(Ellipsis):
                return self.save_reduce(type, (Ellipsis,), obj=obj)
            elif obj is type(NotImplemented):
                return self.save_reduce(type, (NotImplemented,), obj=obj)
            elif obj in _BUILTIN_TYPE_NAMES:
                return self.save_reduce(
                    _builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):  # noqa  # pragma: no branch
                # Parametrized typing constructs in Python < 3.7 are not
                # compatible with type checks and ``isinstance`` semantics. For
                # this reason, it is easier to detect them using a
                # duck-typing-based check (``_is_parametrized_type_hint``) than
                # to populate the Pickler's dispatch with type-specific savers.
                self.save_reduce(
                    _create_parametrized_type_hint,
                    parametrized_type_hint_getinitargs(obj),
                    obj=obj
                )
            elif name is not None:
                Pickler.save_global(self, obj, name=name)
            elif not _should_pickle_by_reference(obj, name=name):
                self._save_reduce_pickle5(*_dynamic_class_reduce(obj), obj=obj)
            else:
                Pickler.save_global(self, obj, name=name)
        dispatch[type] = save_global

        def save_function(self, obj, name=None):
            """ Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
            if _should_pickle_by_reference(obj, name=name):
                return Pickler.save_global(self, obj, name=name)
            elif PYPY and isinstance(obj.__code__, builtin_code_type):
                return self.save_pypy_builtin_func(obj)
            else:
                return self._save_reduce_pickle5(
                    *self._dynamic_function_reduce(obj), obj=obj
                )

        def save_pypy_builtin_func(self, obj):
            """Save pypy equivalent of builtin functions.
            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """
            rv = (types.FunctionType, (obj.__code__, {}, obj.__name__,
                                       obj.__defaults__, obj.__closure__),
                  obj.__dict__)
            self.save_reduce(*rv, obj=obj)

        dispatch[types.FunctionType] = save_function
