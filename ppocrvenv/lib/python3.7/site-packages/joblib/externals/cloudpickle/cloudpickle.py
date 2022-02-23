"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function

import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings

from .compat import pickle
from collections import OrderedDict
from typing import Generic, Union, Tuple, Callable
from pickle import _getattribute
from importlib._bootstrap import _find_spec

try:  # pragma: no branch
    import typing_extensions as _typing_extensions
    from typing_extensions import Literal, Final
except ImportError:
    _typing_extensions = Literal = Final = None

if sys.version_info >= (3, 5, 3):
    from typing import ClassVar
else:  # pragma: no cover
    ClassVar = None

if sys.version_info >= (3, 8):
    from types import CellType
else:
    def f():
        a = 1

        def g():
            return a
        return g
    CellType = type(f().__closure__[0])


# cloudpickle is meant for inter process communication: we expect all
# communicating processes to run the same Python version hence we favor
# communication speed over compatibility:
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Names of modules whose resources should be treated as dynamic.
_PICKLE_BY_VALUE_MODULES = set()

# Track the provenance of reconstructed dynamic classes to make it possible to
# reconstruct instances from the matching singleton class definition when
# appropriate and preserve the usual "isinstance" semantics of Python objects.
_DYNAMIC_CLASS_TRACKER_BY_CLASS = weakref.WeakKeyDictionary()
_DYNAMIC_CLASS_TRACKER_BY_ID = weakref.WeakValueDictionary()
_DYNAMIC_CLASS_TRACKER_LOCK = threading.Lock()

PYPY = platform.python_implementation() == "PyPy"

builtin_code_type = None
if PYPY:
    # builtin-code objects only exist in pypy
    builtin_code_type = type(float.__new__.__code__)

_extract_code_globals_cache = weakref.WeakKeyDictionary()


def _get_or_create_tracker_id(class_def):
    with _DYNAMIC_CLASS_TRACKER_LOCK:
        class_tracker_id = _DYNAMIC_CLASS_TRACKER_BY_CLASS.get(class_def)
        if class_tracker_id is None:
            class_tracker_id = uuid.uuid4().hex
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
            _DYNAMIC_CLASS_TRACKER_BY_ID[class_tracker_id] = class_def
    return class_tracker_id


def _lookup_class_or_track(class_tracker_id, class_def):
    if class_tracker_id is not None:
        with _DYNAMIC_CLASS_TRACKER_LOCK:
            class_def = _DYNAMIC_CLASS_TRACKER_BY_ID.setdefault(
                class_tracker_id, class_def)
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
    return class_def


def register_pickle_by_value(module):
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
    if not isinstance(module, types.ModuleType):
        raise ValueError(
            f"Input should be a module object, got {str(module)} instead"
        )
    # In the future, cloudpickle may need a way to access any module registered
    # for pickling by value in order to introspect relative imports inside
    # functions pickled by value. (see
    # https://github.com/cloudpipe/cloudpickle/pull/417#issuecomment-873684633).
    # This access can be ensured by checking that module is present in
    # sys.modules at registering time and assuming that it will still be in
    # there when accessed during pickling. Another alternative would be to
    # store a weakref to the module. Even though cloudpickle does not implement
    # this introspection yet, in order to avoid a possible breaking change
    # later, we still enforce the presence of module inside sys.modules.
    if module.__name__ not in sys.modules:
        raise ValueError(
            f"{module} was not imported correctly, have you used an "
            f"`import` statement to access it?"
        )
    _PICKLE_BY_VALUE_MODULES.add(module.__name__)


def unregister_pickle_by_value(module):
    """Unregister that the input module should be pickled by value."""
    if not isinstance(module, types.ModuleType):
        raise ValueError(
            f"Input should be a module object, got {str(module)} instead"
        )
    if module.__name__ not in _PICKLE_BY_VALUE_MODULES:
        raise ValueError(f"{module} is not registered for pickle by value")
    else:
        _PICKLE_BY_VALUE_MODULES.remove(module.__name__)


def list_registry_pickle_by_value():
    return _PICKLE_BY_VALUE_MODULES.copy()


def _is_registered_pickle_by_value(module):
    module_name = module.__name__
    if module_name in _PICKLE_BY_VALUE_MODULES:
        return True
    while True:
        parent_name = module_name.rsplit(".", 1)[0]
        if parent_name == module_name:
            break
        if parent_name in _PICKLE_BY_VALUE_MODULES:
            return True
        module_name = parent_name
    return False


def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    if sys.version_info[:2] < (3, 7) and isinstance(obj, typing.TypeVar):  # pragma: no branch  # noqa
        # Workaround bug in old Python versions: prior to Python 3.7,
        # T.__module__ would always be set to "typing" even when the TypeVar T
        # would be defined in a different module.
        if name is not None and getattr(typing, name, None) is obj:
            # Built-in TypeVar defined in typing such as AnyStr
            return 'typing'
        else:
            # User defined or third-party TypeVar: __module__ attribute is
            # irrelevant, thus trigger a exhaustive search for obj in all
            # modules.
            module_name = None
    else:
        module_name = getattr(obj, '__module__', None)

    if module_name is not None:
        return module_name
    # Protect the iteration by using a copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr or
    # other threads importing at the same time.
    for module_name, module in sys.modules.copy().items():
        # Some modules such as coverage can inject non-module objects inside
        # sys.modules
        if (
                module_name == '__main__' or
                module is None or
                not isinstance(module, types.ModuleType)
        ):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None


def _should_pickle_by_reference(obj, name=None):
    """Test whether an function or a class should be pickled by reference

     Pickling by reference means by that the object (typically a function or a
     class) is an attribute of a module that is assumed to be importable in the
     target Python environment. Loading will therefore rely on importing the
     module and then calling `getattr` on it to access the function or class.

     Pickling by reference is the only option to pickle functions and classes
     in the standard library. In cloudpickle the alternative option is to
     pickle by value (for instance for interactively or locally defined
     functions and classes or for attributes of modules that have been
     explicitly registered to be pickled by value.
     """
    if isinstance(obj, types.FunctionType) or issubclass(type(obj), type):
        module_and_name = _lookup_module_and_qualname(obj, name=name)
        if module_and_name is None:
            return False
        module, name = module_and_name
        return not _is_registered_pickle_by_value(module)

    elif isinstance(obj, types.ModuleType):
        # We assume that sys.modules is primarily used as a cache mechanism for
        # the Python import machinery. Checking if a module has been added in
        # is sys.modules therefore a cheap and simple heuristic to tell us
        # whether we can assume that a given module could be imported by name
        # in another Python process.
        if _is_registered_pickle_by_value(obj):
            return False
        return obj.__name__ in sys.modules
    else:
        raise TypeError(
            "cannot check importability of {} instances".format(
                type(obj).__name__)
        )


def _lookup_module_and_qualname(obj, name=None):
    if name is None:
        name = getattr(obj, '__qualname__', None)
    if name is None:  # pragma: no cover
        # This used to be needed for Python 2.7 support but is probably not
        # needed anymore. However we keep the __name__ introspection in case
        # users of cloudpickle rely on this old behavior for unknown reasons.
        name = getattr(obj, '__name__', None)

    module_name = _whichmodule(obj, name)

    if module_name is None:
        # In this case, obj.__module__ is None AND obj was not found in any
        # imported module. obj is thus treated as dynamic.
        return None

    if module_name == "__main__":
        return None

    # Note: if module_name is in sys.modules, the corresponding module is
    # assumed importable at unpickling time. See #357
    module = sys.modules.get(module_name, None)
    if module is None:
        # The main reason why obj's module would not be imported is that this
        # module has been dynamically created, using for example
        # types.ModuleType. The other possibility is that module was removed
        # from sys.modules after obj was created/imported. But this case is not
        # supported, as the standard pickle does not support it either.
        return None

    try:
        obj2, parent = _getattribute(module, name)
    except AttributeError:
        # obj was not found inside the module it points to
        return None
    if obj2 is not obj:
        return None
    return module, name


def _extract_code_globals(co):
    """
    Find all globals names read or written to by codeblock co
    """
    out_names = _extract_code_globals_cache.get(co)
    if out_names is None:
        names = co.co_names
        # We use a dict with None values instead of a set to get a
        # deterministic order (assuming Python 3.6+) and avoid introducing
        # non-deterministic pickle bytes as a results.
        out_names = {names[oparg]: None for _, oparg in _walk_global_ops(co)}

        # Declaring a function inside another one using the "def ..."
        # syntax generates a constant code object corresponding to the one
        # of the nested function's As the nested function may itself need
        # global variables, we need to introspect its code, extract its
        # globals, (look for code object in it's co_consts attribute..) and
        # add the result to code_globals
        if co.co_consts:
            for const in co.co_consts:
                if isinstance(const, types.CodeType):
                    out_names.update(_extract_code_globals(const))

        _extract_code_globals_cache[co] = out_names

    return out_names


def _find_imported_submodules(code, top_level_dependencies):
    """
    Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """

    subimports = []
    # check if any known dependency is an imported package
    for x in top_level_dependencies:
        if (isinstance(x, types.ModuleType) and
                hasattr(x, '__package__') and x.__package__):
            # check if the package has any currently loaded sub-imports
            prefix = x.__name__ + '.'
            # A concurrent thread could mutate sys.modules,
            # make sure we iterate over a copy to avoid exceptions
            for name in list(sys.modules):
                # Older versions of pytest will add a "None" module to
                # sys.modules.
                if name is not None and name.startswith(prefix):
                    # check whether the function can address the sub-module
                    tokens = set(name[len(prefix):].split('.'))
                    if not tokens - set(code.co_names):
                        subimports.append(sys.modules[name])
    return subimports


def cell_set(cell, value):
    """Set the value of a closure cell.

    The point of this function is to set the cell_contents attribute of a cell
    after its creation. This operation is necessary in case the cell contains a
    reference to the function the cell belongs to, as when calling the
    function's constructor
    ``f = types.FunctionType(code, globals, name, argdefs, closure)``,
    closure will not be able to contain the yet-to-be-created f.

    In Python3.7, cell_contents is writeable, so setting the contents of a cell
    can be done simply using
    >>> cell.cell_contents = value

    In earlier Python3 versions, the cell_contents attribute of a cell is read
    only, but this limitation can be worked around by leveraging the Python 3
    ``nonlocal`` keyword.

    In Python2 however, this attribute is read only, and there is no
    ``nonlocal`` keyword. For this reason, we need to come up with more
    complicated hacks to set this attribute.

    The chosen approach is to create a function with a STORE_DEREF opcode,
    which sets the content of a closure variable. Typically:

    >>> def inner(value):
    ...     lambda: cell  # the lambda makes cell a closure
    ...     cell = value  # cell is a closure, so this triggers a STORE_DEREF

    (Note that in Python2, A STORE_DEREF can never be triggered from an inner
    function. The function g for example here
    >>> def f(var):
    ...     def g():
    ...         var += 1
    ...     return g

    will not modify the closure variable ``var```inplace, but instead try to
    load a local variable var and increment it. As g does not assign the local
    variable ``var`` any initial value, calling f(1)() will fail at runtime.)

    Our objective is to set the value of a given cell ``cell``. So we need to
    somewhat reference our ``cell`` object into the ``inner`` function so that
    this object (and not the smoke cell of the lambda function) gets affected
    by the STORE_DEREF operation.

    In inner, ``cell`` is referenced as a cell variable (an enclosing variable
    that is referenced by the inner function). If we create a new function
    cell_set with the exact same code as ``inner``, but with ``cell`` marked as
    a free variable instead, the STORE_DEREF will be applied on its closure -
    ``cell``, which we can specify explicitly during construction! The new
    cell_set variable thus actually sets the contents of a specified cell!

    Note: we do not make use of the ``nonlocal`` keyword to set the contents of
    a cell in early python3 versions to limit possible syntax errors in case
    test and checker libraries decide to parse the whole file.
    """

    if sys.version_info[:2] >= (3, 7):  # pragma: no branch
        cell.cell_contents = value
    else:
        _cell_set = types.FunctionType(
            _cell_set_template_code, {}, '_cell_set', (), (cell,),)
        _cell_set(value)


def _make_cell_set_template_code():
    def _cell_set_factory(value):
        lambda: cell
        cell = value

    co = _cell_set_factory.__code__

    _cell_set_template_code = types.CodeType(
        co.co_argcount,
        co.co_kwonlyargcount,   # Python 3 only argument
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_cellvars,  # co_freevars is initialized with co_cellvars
        (),  # co_cellvars is made empty
    )
    return _cell_set_template_code


if sys.version_info[:2] < (3, 7):
    _cell_set_template_code = _make_cell_set_template_code()

# relevant opcodes
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG


_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


def _builtin_type(name):
    if name == "ClassType":  # pragma: no cover
        # Backward compat to load pickle files generated with cloudpickle
        # < 1.3 even if loading pickle files from older versions is not
        # officially supported.
        return type
    return getattr(types, name)


def _walk_global_ops(code):
    """
    Yield (opcode, argument number) tuples for all
    global-referencing instructions in *code*.
    """
    for instr in dis.get_instructions(code):
        op = instr.opcode
        if op in GLOBAL_OPS:
            yield op, instr.arg


def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited methods"""
    clsdict = dict(cls.__dict__)  # copy dict proxy to a dict
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)
    to_remove = []
    for name, value in clsdict.items():
        try:
            base_value = inherited_dict[name]
            if value is base_value:
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        clsdict.pop(name)
    return clsdict


if sys.version_info[:2] < (3, 7):  # pragma: no branch
    def _is_parametrized_type_hint(obj):
        # This is very cheap but might generate false positives. So try to
        # narrow it down is good as possible.
        type_module = getattr(type(obj), '__module__', None)
        from_typing_extensions = type_module == 'typing_extensions'
        from_typing = type_module == 'typing'

        # general typing Constructs
        is_typing = getattr(obj, '__origin__', None) is not None

        # typing_extensions.Literal
        is_literal = (
            (getattr(obj, '__values__', None) is not None)
            and from_typing_extensions
        )

        # typing_extensions.Final
        is_final = (
            (getattr(obj, '__type__', None) is not None)
            and from_typing_extensions
        )

        # typing.ClassVar
        is_classvar = (
            (getattr(obj, '__type__', None) is not None) and from_typing
        )

        # typing.Union/Tuple for old Python 3.5
        is_union = getattr(obj, '__union_params__', None) is not None
        is_tuple = getattr(obj, '__tuple_params__', None) is not None
        is_callable = (
            getattr(obj, '__result__', None) is not None and
            getattr(obj, '__args__', None) is not None
        )
        return any((is_typing, is_literal, is_final, is_classvar, is_union,
                    is_tuple, is_callable))

    def _create_parametrized_type_hint(origin, args):
        return origin[args]
else:
    _is_parametrized_type_hint = None
    _create_parametrized_type_hint = None


def parametrized_type_hint_getinitargs(obj):
    # The distorted type check sematic for typing construct becomes:
    # ``type(obj) is type(TypeHint)``, which means "obj is a
    # parametrized TypeHint"
    if type(obj) is type(Literal):  # pragma: no branch
        initargs = (Literal, obj.__values__)
    elif type(obj) is type(Final):  # pragma: no branch
        initargs = (Final, obj.__type__)
    elif type(obj) is type(ClassVar):
        initargs = (ClassVar, obj.__type__)
    elif type(obj) is type(Generic):
        parameters = obj.__parameters__
        if len(obj.__parameters__) > 0:
            # in early Python 3.5, __parameters__ was sometimes
            # preferred to __args__
            initargs = (obj.__origin__, parameters)

        else:
            initargs = (obj.__origin__, obj.__args__)
    elif type(obj) is type(Union):
        if sys.version_info < (3, 5, 3):  # pragma: no cover
            initargs = (Union, obj.__union_params__)
        else:
            initargs = (Union, obj.__args__)
    elif type(obj) is type(Tuple):
        if sys.version_info < (3, 5, 3):  # pragma: no cover
            initargs = (Tuple, obj.__tuple_params__)
        else:
            initargs = (Tuple, obj.__args__)
    elif type(obj) is type(Callable):
        if sys.version_info < (3, 5, 3):  # pragma: no cover
            args = obj.__args__
            result = obj.__result__
            if args != Ellipsis:
                if isinstance(args, tuple):
                    args = list(args)
                else:
                    args = [args]
        else:
            (*args, result) = obj.__args__
            if len(args) == 1 and args[0] is Ellipsis:
                args = Ellipsis
            else:
                args = list(args)
        initargs = (Callable, (args, result))
    else:  # pragma: no cover
        raise pickle.PicklingError(
            "Cloudpickle Error: Unknown type {}".format(type(obj))
        )
    return initargs


# Tornado support

def is_tornado_coroutine(func):
    """
    Return whether *func* is a Tornado coroutine function.
    Running coroutines are not supported.
    """
    if 'tornado.gen' not in sys.modules:
        return False
    gen = sys.modules['tornado.gen']
    if not hasattr(gen, "is_coroutine_function"):
        # Tornado version is too old
        return False
    return gen.is_coroutine_function(func)


def _rebuild_tornado_coroutine(func):
    from tornado import gen
    return gen.coroutine(func)


# including pickles unloading functions in this namespace
load = pickle.load
loads = pickle.loads


def subimport(name):
    # We cannot do simply: `return __import__(name)`: Indeed, if ``name`` is
    # the name of a submodule, __import__ will return the top-level root module
    # of this submodule. For instance, __import__('os.path') returns the `os`
    # module.
    __import__(name)
    return sys.modules[name]


def dynamic_subimport(name, vars):
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    mod.__dict__['__builtins__'] = builtins.__dict__
    return mod


def _gen_ellipsis():
    return Ellipsis


def _gen_not_implemented():
    return NotImplemented


def _get_cell_contents(cell):
    try:
        return cell.cell_contents
    except ValueError:
        # sentinel used by ``_fill_function`` which will leave the cell empty
        return _empty_cell_value


def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    return cls()


@instance
class _empty_cell_value(object):
    """sentinel for empty closures
    """
    @classmethod
    def __reduce__(cls):
        return cls.__name__


def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    if len(args) == 2:
        func = args[0]
        state = args[1]
    elif len(args) == 5:
        # Backwards compat for cloudpickle v0.4.0, after which the `module`
        # argument was introduced
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'closure_values']
        state = dict(zip(keys, args[1:]))
    elif len(args) == 6:
        # Backwards compat for cloudpickle v0.4.1, after which the function
        # state was passed as a dict to the _fill_function it-self.
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'module', 'closure_values']
        state = dict(zip(keys, args[1:]))
    else:
        raise ValueError('Unexpected _fill_value arguments: %r' % (args,))

    # - At pickling time, any dynamic global variable used by func is
    #   serialized by value (in state['globals']).
    # - At unpickling time, func's __globals__ attribute is initialized by
    #   first retrieving an empty isolated namespace that will be shared
    #   with other functions pickled from the same original module
    #   by the same CloudPickler instance and then updated with the
    #   content of state['globals'] to populate the shared isolated
    #   namespace with all the global variables that are specifically
    #   referenced for this function.
    func.__globals__.update(state['globals'])

    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    if 'annotations' in state:
        func.__annotations__ = state['annotations']
    if 'doc' in state:
        func.__doc__ = state['doc']
    if 'name' in state:
        func.__name__ = state['name']
    if 'module' in state:
        func.__module__ = state['module']
    if 'qualname' in state:
        func.__qualname__ = state['qualname']
    if 'kwdefaults' in state:
        func.__kwdefaults__ = state['kwdefaults']
    # _cloudpickle_subimports is a set of submodules that must be loaded for
    # the pickled function to work correctly at unpickling time. Now that these
    # submodules are depickled (hence imported), they can be removed from the
    # object's state (the object state only served as a reference holder to
    # these submodules)
    if '_cloudpickle_submodules' in state:
        state.pop('_cloudpickle_submodules')

    cells = func.__closure__
    if cells is not None:
        for cell, value in zip(cells, state['closure_values']):
            if value is not _empty_cell_value:
                cell_set(cell, value)

    return func


def _make_empty_cell():
    if False:
        # trick the compiler into creating an empty cell in our lambda
        cell = None
        raise AssertionError('this route should not be executed')

    return (lambda: cell).__closure__[0]


def _make_cell(value=_empty_cell_value):
    cell = _make_empty_cell()
    if value is not _empty_cell_value:
        cell_set(cell, value)
    return cell


def _make_skel_func(code, cell_count, base_globals=None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    # This function is deprecated and should be removed in cloudpickle 1.7
    warnings.warn(
        "A pickle file created using an old (<=1.4.1) version of cloudpickle "
        "is currently being loaded. This is not supported by cloudpickle and "
        "will break in cloudpickle 1.7", category=UserWarning
    )
    # This is backward-compatibility code: for cloudpickle versions between
    # 0.5.4 and 0.7, base_globals could be a string or None. base_globals
    # should now always be a dictionary.
    if base_globals is None or isinstance(base_globals, str):
        base_globals = {}

    base_globals['__builtins__'] = __builtins__

    closure = (
        tuple(_make_empty_cell() for _ in range(cell_count))
        if cell_count >= 0 else
        None
    )
    return types.FunctionType(code, base_globals, None, None, closure)


def _make_skeleton_class(type_constructor, name, bases, type_kwargs,
                         class_tracker_id, extra):
    """Build dynamic class with an empty __dict__ to be filled once memoized

    If class_tracker_id is not None, try to lookup an existing class definition
    matching that id. If none is found, track a newly reconstructed class
    definition under that id so that other instances stemming from the same
    class id will also reuse this class definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    skeleton_class = types.new_class(
        name, bases, {'metaclass': type_constructor},
        lambda ns: ns.update(type_kwargs)
    )
    return _lookup_class_or_track(class_tracker_id, skeleton_class)


def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    registry = None
    for attrname, attr in class_dict.items():
        if attrname == "_abc_impl":
            registry = attr
        else:
            setattr(skeleton_class, attrname, attr)
    if registry is not None:
        for subclass in registry:
            skeleton_class.register(subclass)

    return skeleton_class


def _make_skeleton_enum(bases, name, qualname, members, module,
                        class_tracker_id, extra):
    """Build dynamic enum with an empty __dict__ to be filled once memoized

    The creation of the enum class is inspired by the code of
    EnumMeta._create_.

    If class_tracker_id is not None, try to lookup an existing enum definition
    matching that id. If none is found, track a newly reconstructed enum
    definition under that id so that other instances stemming from the same
    class id will also reuse this enum definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    # enums always inherit from their base Enum class at the last position in
    # the list of base classes:
    enum_base = bases[-1]
    metacls = enum_base.__class__
    classdict = metacls.__prepare__(name, bases)

    for member_name, member_value in members.items():
        classdict[member_name] = member_value
    enum_class = metacls.__new__(metacls, name, bases, classdict)
    enum_class.__module__ = module
    enum_class.__qualname__ = qualname

    return _lookup_class_or_track(class_tracker_id, enum_class)


def _make_typevar(name, bound, constraints, covariant, contravariant,
                  class_tracker_id):
    tv = typing.TypeVar(
        name, *constraints, bound=bound,
        covariant=covariant, contravariant=contravariant
    )
    if class_tracker_id is not None:
        return _lookup_class_or_track(class_tracker_id, tv)
    else:  # pragma: nocover
        # Only for Python 3.5.3 compat.
        return tv


def _decompose_typevar(obj):
    try:
        class_tracker_id = _get_or_create_tracker_id(obj)
    except TypeError:  # pragma: nocover
        # TypeVar instances are not weakref-able in Python 3.5.3
        class_tracker_id = None
    return (
        obj.__name__, obj.__bound__, obj.__constraints__,
        obj.__covariant__, obj.__contravariant__,
        class_tracker_id,
    )


def _typevar_reduce(obj):
    # TypeVar instances require the module information hence why we
    # are not using the _should_pickle_by_reference directly
    module_and_name = _lookup_module_and_qualname(obj, name=obj.__name__)

    if module_and_name is None:
        return (_make_typevar, _decompose_typevar(obj))
    elif _is_registered_pickle_by_value(module_and_name[0]):
        return (_make_typevar, _decompose_typevar(obj))

    return (getattr, module_and_name)


def _get_bases(typ):
    if hasattr(typ, '__orig_bases__'):
        # For generic types (see PEP 560)
        bases_attr = '__orig_bases__'
    else:
        # For regular class objects
        bases_attr = '__bases__'
    return getattr(typ, bases_attr)


def _make_dict_keys(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict.fromkeys(obj).keys()
    else:
        return dict.fromkeys(obj).keys()


def _make_dict_values(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict((i, _) for i, _ in enumerate(obj)).values()
    else:
        return {i: _ for i, _ in enumerate(obj)}.values()


def _make_dict_items(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict(obj).items()
    else:
        return obj.items()
