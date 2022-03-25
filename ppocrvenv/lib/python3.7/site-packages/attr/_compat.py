# SPDX-License-Identifier: MIT

from __future__ import absolute_import, division, print_function

import platform
import sys
import threading
import types
import warnings


PY2 = sys.version_info[0] == 2
PYPY = platform.python_implementation() == "PyPy"
PY36 = sys.version_info[:2] >= (3, 6)
HAS_F_STRINGS = PY36
PY310 = sys.version_info[:2] >= (3, 10)


if PYPY or PY36:
    ordered_dict = dict
else:
    from collections import OrderedDict

    ordered_dict = OrderedDict


if PY2:
    from collections import Mapping, Sequence

    from UserDict import IterableUserDict

    # We 'bundle' isclass instead of using inspect as importing inspect is
    # fairly expensive (order of 10-15 ms for a modern machine in 2016)
    def isclass(klass):
        return isinstance(klass, (type, types.ClassType))

    def new_class(name, bases, kwds, exec_body):
        """
        A minimal stub of types.new_class that we need for make_class.
        """
        ns = {}
        exec_body(ns)

        return type(name, bases, ns)

    # TYPE is used in exceptions, repr(int) is different on Python 2 and 3.
    TYPE = "type"

    def iteritems(d):
        return d.iteritems()

    # Python 2 is bereft of a read-only dict proxy, so we make one!
    class ReadOnlyDict(IterableUserDict):
        """
        Best-effort read-only dict wrapper.
        """

        def __setitem__(self, key, val):
            # We gently pretend we're a Python 3 mappingproxy.
            raise TypeError(
                "'mappingproxy' object does not support item assignment"
            )

        def update(self, _):
            # We gently pretend we're a Python 3 mappingproxy.
            raise AttributeError(
                "'mappingproxy' object has no attribute 'update'"
            )

        def __delitem__(self, _):
            # We gently pretend we're a Python 3 mappingproxy.
            raise TypeError(
                "'mappingproxy' object does not support item deletion"
            )

        def clear(self):
            # We gently pretend we're a Python 3 mappingproxy.
            raise AttributeError(
                "'mappingproxy' object has no attribute 'clear'"
            )

        def pop(self, key, default=None):
            # We gently pretend we're a Python 3 mappingproxy.
            raise AttributeError(
                "'mappingproxy' object has no attribute 'pop'"
            )

        def popitem(self):
            # We gently pretend we're a Python 3 mappingproxy.
            raise AttributeError(
                "'mappingproxy' object has no attribute 'popitem'"
            )

        def setdefault(self, key, default=None):
            # We gently pretend we're a Python 3 mappingproxy.
            raise AttributeError(
                "'mappingproxy' object has no attribute 'setdefault'"
            )

        def __repr__(self):
            # Override to be identical to the Python 3 version.
            return "mappingproxy(" + repr(self.data) + ")"

    def metadata_proxy(d):
        res = ReadOnlyDict()
        res.data.update(d)  # We blocked update, so we have to do it like this.
        return res

    def just_warn(*args, **kw):  # pragma: no cover
        """
        We only warn on Python 3 because we are not aware of any concrete
        consequences of not setting the cell on Python 2.
        """

else:  # Python 3 and later.
    from collections.abc import Mapping, Sequence  # noqa

    def just_warn(*args, **kw):
        """
        We only warn on Python 3 because we are not aware of any concrete
        consequences of not setting the cell on Python 2.
        """
        warnings.warn(
            "Running interpreter doesn't sufficiently support code object "
            "introspection.  Some features like bare super() or accessing "
            "__class__ will not work with slotted classes.",
            RuntimeWarning,
            stacklevel=2,
        )

    def isclass(klass):
        return isinstance(klass, type)

    TYPE = "class"

    def iteritems(d):
        return d.items()

    new_class = types.new_class

    def metadata_proxy(d):
        return types.MappingProxyType(dict(d))


def make_set_closure_cell():
    """Return a function of two arguments (cell, value) which sets
    the value stored in the closure cell `cell` to `value`.
    """
    # pypy makes this easy. (It also supports the logic below, but
    # why not do the easy/fast thing?)
    if PYPY:

        def set_closure_cell(cell, value):
            cell.__setstate__((value,))

        return set_closure_cell

    # Otherwise gotta do it the hard way.

    # Create a function that will set its first cellvar to `value`.
    def set_first_cellvar_to(value):
        x = value
        return

        # This function will be eliminated as dead code, but
        # not before its reference to `x` forces `x` to be
        # represented as a closure cell rather than a local.
        def force_x_to_be_a_cell():  # pragma: no cover
            return x

    try:
        # Extract the code object and make sure our assumptions about
        # the closure behavior are correct.
        if PY2:
            co = set_first_cellvar_to.func_code
        else:
            co = set_first_cellvar_to.__code__
        if co.co_cellvars != ("x",) or co.co_freevars != ():
            raise AssertionError  # pragma: no cover

        # Convert this code object to a code object that sets the
        # function's first _freevar_ (not cellvar) to the argument.
        if sys.version_info >= (3, 8):
            # CPython 3.8+ has an incompatible CodeType signature
            # (added a posonlyargcount argument) but also added
            # CodeType.replace() to do this without counting parameters.
            set_first_freevar_code = co.replace(
                co_cellvars=co.co_freevars, co_freevars=co.co_cellvars
            )
        else:
            args = [co.co_argcount]
            if not PY2:
                args.append(co.co_kwonlyargcount)
            args.extend(
                [
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
                    # These two arguments are reversed:
                    co.co_cellvars,
                    co.co_freevars,
                ]
            )
            set_first_freevar_code = types.CodeType(*args)

        def set_closure_cell(cell, value):
            # Create a function using the set_first_freevar_code,
            # whose first closure cell is `cell`. Calling it will
            # change the value of that cell.
            setter = types.FunctionType(
                set_first_freevar_code, {}, "setter", (), (cell,)
            )
            # And call it to set the cell.
            setter(value)

        # Make sure it works on this interpreter:
        def make_func_with_cell():
            x = None

            def func():
                return x  # pragma: no cover

            return func

        if PY2:
            cell = make_func_with_cell().func_closure[0]
        else:
            cell = make_func_with_cell().__closure__[0]
        set_closure_cell(cell, 100)
        if cell.cell_contents != 100:
            raise AssertionError  # pragma: no cover

    except Exception:
        return just_warn
    else:
        return set_closure_cell


set_closure_cell = make_set_closure_cell()

# Thread-local global to track attrs instances which are already being repr'd.
# This is needed because there is no other (thread-safe) way to pass info
# about the instances that are already being repr'd through the call stack
# in order to ensure we don't perform infinite recursion.
#
# For instance, if an instance contains a dict which contains that instance,
# we need to know that we're already repr'ing the outside instance from within
# the dict's repr() call.
#
# This lives here rather than in _make.py so that the functions in _make.py
# don't have a direct reference to the thread-local in their globals dict.
# If they have such a reference, it breaks cloudpickle.
repr_context = threading.local()
