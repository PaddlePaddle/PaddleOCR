__all__ = ('ForwardRefPolicy', 'TypeHintWarning', 'typechecked', 'check_return_type',
           'check_argument_types', 'check_type', 'TypeWarning', 'TypeChecker',
           'typeguard_ignore')

import collections.abc
import gc
import inspect
import sys
import threading
from collections import OrderedDict
from enum import Enum
from functools import partial, wraps
from inspect import Parameter, isclass, isfunction, isgeneratorfunction
from io import BufferedIOBase, IOBase, RawIOBase, TextIOBase
from traceback import extract_stack, print_stack
from types import CodeType, FunctionType
from typing import (
    IO, TYPE_CHECKING, AbstractSet, Any, AsyncIterable, AsyncIterator, BinaryIO, Callable, Dict,
    Generator, Iterable, Iterator, List, NewType, Optional, Sequence, Set, TextIO, Tuple, Type,
    TypeVar, Union, get_type_hints, overload)
from unittest.mock import Mock
from warnings import warn
from weakref import WeakKeyDictionary, WeakValueDictionary

# Python 3.8+
try:
    from typing_extensions import Literal
except ImportError:
    try:
        from typing import Literal
    except ImportError:
        Literal = None

# Python 3.5.4+ / 3.6.2+
try:
    from typing_extensions import NoReturn
except ImportError:
    try:
        from typing import NoReturn
    except ImportError:
        NoReturn = None

# Python 3.6+
try:
    from inspect import isasyncgen, isasyncgenfunction
    from typing import AsyncGenerator
except ImportError:
    AsyncGenerator = None

    def isasyncgen(obj):
        return False

    def isasyncgenfunction(func):
        return False

# Python 3.8+
try:
    from typing import ForwardRef
    evaluate_forwardref = ForwardRef._evaluate
except ImportError:
    from typing import _ForwardRef as ForwardRef
    evaluate_forwardref = ForwardRef._eval_type

if sys.version_info >= (3, 10):
    from typing import is_typeddict
else:
    _typed_dict_meta_types = ()
    if sys.version_info >= (3, 8):
        from typing import _TypedDictMeta
        _typed_dict_meta_types += (_TypedDictMeta,)

    try:
        from typing_extensions import _TypedDictMeta
        _typed_dict_meta_types += (_TypedDictMeta,)
    except ImportError:
        pass

    def is_typeddict(tp) -> bool:
        return isinstance(tp, _typed_dict_meta_types)


if TYPE_CHECKING:
    _F = TypeVar("_F")

    def typeguard_ignore(f: _F) -> _F:
        """This decorator is a noop during static type-checking."""
        return f
else:
    from typing import no_type_check as typeguard_ignore


_type_hints_map = WeakKeyDictionary()  # type: Dict[FunctionType, Dict[str, Any]]
_functions_map = WeakValueDictionary()  # type: Dict[CodeType, FunctionType]
_missing = object()

T_CallableOrType = TypeVar('T_CallableOrType', bound=Callable[..., Any])

# Lifted from mypy.sharedparse
BINARY_MAGIC_METHODS = {
    "__add__",
    "__and__",
    "__cmp__",
    "__divmod__",
    "__div__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__iadd__",
    "__iand__",
    "__idiv__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__or__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rdiv__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__xor__",
}


class ForwardRefPolicy(Enum):
    """Defines how unresolved forward references are handled."""

    ERROR = 1  #: propagate the :exc:`NameError` from :func:`~typing.get_type_hints`
    WARN = 2  #: remove the annotation and emit a TypeHintWarning
    #: replace the annotation with the argument's class if the qualified name matches, else remove
    #: the annotation
    GUESS = 3


class TypeHintWarning(UserWarning):
    """
    A warning that is emitted when a type hint in string form could not be resolved to an actual
    type.
    """


class _TypeCheckMemo:
    __slots__ = 'globals', 'locals'

    def __init__(self, globals: Dict[str, Any], locals: Dict[str, Any]):
        self.globals = globals
        self.locals = locals


def _strip_annotation(annotation):
    if isinstance(annotation, str):
        return annotation.strip("'")
    else:
        return annotation


class _CallMemo(_TypeCheckMemo):
    __slots__ = 'func', 'func_name', 'arguments', 'is_generator', 'type_hints'

    def __init__(self, func: Callable, frame_locals: Optional[Dict[str, Any]] = None,
                 args: tuple = None, kwargs: Dict[str, Any] = None,
                 forward_refs_policy=ForwardRefPolicy.ERROR):
        super().__init__(func.__globals__, frame_locals)
        self.func = func
        self.func_name = function_name(func)
        self.is_generator = isgeneratorfunction(func)
        signature = inspect.signature(func)

        if args is not None and kwargs is not None:
            self.arguments = signature.bind(*args, **kwargs).arguments
        else:
            assert frame_locals is not None, 'frame must be specified if args or kwargs is None'
            self.arguments = frame_locals

        self.type_hints = _type_hints_map.get(func)
        if self.type_hints is None:
            while True:
                if sys.version_info < (3, 5, 3):
                    frame_locals = dict(frame_locals)

                try:
                    hints = get_type_hints(func, localns=frame_locals)
                except NameError as exc:
                    if forward_refs_policy is ForwardRefPolicy.ERROR:
                        raise

                    typename = str(exc).split("'", 2)[1]
                    for param in signature.parameters.values():
                        if _strip_annotation(param.annotation) == typename:
                            break
                    else:
                        raise

                    func_name = function_name(func)
                    if forward_refs_policy is ForwardRefPolicy.GUESS:
                        if param.name in self.arguments:
                            argtype = self.arguments[param.name].__class__
                            stripped = _strip_annotation(param.annotation)
                            if stripped == argtype.__qualname__:
                                func.__annotations__[param.name] = argtype
                                msg = ('Replaced forward declaration {!r} in {} with {!r}'
                                       .format(stripped, func_name, argtype))
                                warn(TypeHintWarning(msg))
                                continue

                    msg = 'Could not resolve type hint {!r} on {}: {}'.format(
                        param.annotation, function_name(func), exc)
                    warn(TypeHintWarning(msg))
                    del func.__annotations__[param.name]
                else:
                    break

            self.type_hints = OrderedDict()
            for name, parameter in signature.parameters.items():
                if name in hints:
                    annotated_type = hints[name]

                    # PEP 428 discourages it by MyPy does not complain
                    if parameter.default is None:
                        annotated_type = Optional[annotated_type]

                    if parameter.kind == Parameter.VAR_POSITIONAL:
                        self.type_hints[name] = Tuple[annotated_type, ...]
                    elif parameter.kind == Parameter.VAR_KEYWORD:
                        self.type_hints[name] = Dict[str, annotated_type]
                    else:
                        self.type_hints[name] = annotated_type

            if 'return' in hints:
                self.type_hints['return'] = hints['return']

            _type_hints_map[func] = self.type_hints


def resolve_forwardref(maybe_ref, memo: _TypeCheckMemo):
    if isinstance(maybe_ref, ForwardRef):
        if sys.version_info < (3, 9, 0):
            return evaluate_forwardref(maybe_ref, memo.globals, memo.locals)
        else:
            return evaluate_forwardref(maybe_ref, memo.globals, memo.locals, frozenset())

    else:
        return maybe_ref


def get_type_name(type_):
    name = (getattr(type_, '__name__', None) or getattr(type_, '_name', None) or
            getattr(type_, '__forward_arg__', None))
    if name is None:
        origin = getattr(type_, '__origin__', None)
        name = getattr(origin, '_name', None)
        if name is None and not inspect.isclass(type_):
            name = type_.__class__.__name__.strip('_')

    args = getattr(type_, '__args__', ()) or getattr(type_, '__values__', ())
    if args != getattr(type_, '__parameters__', ()):
        if name == 'Literal':
            formatted_args = ', '.join(str(arg) for arg in args)
        else:
            formatted_args = ', '.join(get_type_name(arg) for arg in args)

        name = '{}[{}]'.format(name, formatted_args)

    module = getattr(type_, '__module__', None)
    if module not in (None, 'typing', 'typing_extensions', 'builtins'):
        name = module + '.' + name

    return name


def find_function(frame) -> Optional[Callable]:
    """
    Return a function object from the garbage collector that matches the frame's code object.

    This process is unreliable as several function objects could use the same code object.
    Fortunately the likelihood of this happening with the combination of the function objects
    having different type annotations is a very rare occurrence.

    :param frame: a frame object
    :return: a function object if one was found, ``None`` if not

    """
    func = _functions_map.get(frame.f_code)
    if func is None:
        for obj in gc.get_referrers(frame.f_code):
            if inspect.isfunction(obj):
                if func is None:
                    # The first match was found
                    func = obj
                else:
                    # A second match was found
                    return None

        # Cache the result for future lookups
        if func is not None:
            _functions_map[frame.f_code] = func
        else:
            raise LookupError('target function not found')

    return func


def qualified_name(obj) -> str:
    """
    Return the qualified name (e.g. package.module.Type) for the given object.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    type_ = obj if inspect.isclass(obj) else type(obj)
    module = type_.__module__
    qualname = type_.__qualname__
    return qualname if module in ('typing', 'builtins') else '{}.{}'.format(module, qualname)


def function_name(func: Callable) -> str:
    """
    Return the qualified name of the given function.

    Builtins and types from the :mod:`typing` package get special treatment by having the module
    name stripped from the generated name.

    """
    # For partial functions and objects with __call__ defined, __qualname__ does not exist
    # For functions run in `exec` with a custom namespace, __module__ can be None
    module = getattr(func, '__module__', '') or ''
    qualname = (module + '.') if module not in ('builtins', '') else ''
    return qualname + getattr(func, '__qualname__', repr(func))


def check_callable(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not callable(value):
        raise TypeError('{} must be a callable'.format(argname))

    if getattr(expected_type, "__args__", None):
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return

        if hasattr(expected_type, '__result__'):
            # Python 3.5
            argument_types = expected_type.__args__
            check_args = argument_types is not Ellipsis
        else:
            # Python 3.6
            argument_types = expected_type.__args__[:-1]
            check_args = argument_types != (Ellipsis,)

        if check_args:
            # The callable must not have keyword-only arguments without defaults
            unfulfilled_kwonlyargs = [
                param.name for param in signature.parameters.values() if
                param.kind == Parameter.KEYWORD_ONLY and param.default == Parameter.empty]
            if unfulfilled_kwonlyargs:
                raise TypeError(
                    'callable passed as {} has mandatory keyword-only arguments in its '
                    'declaration: {}'.format(argname, ', '.join(unfulfilled_kwonlyargs)))

            num_mandatory_args = len([
                param.name for param in signature.parameters.values()
                if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD) and
                param.default is Parameter.empty])
            has_varargs = any(param for param in signature.parameters.values()
                              if param.kind == Parameter.VAR_POSITIONAL)

            if num_mandatory_args > len(argument_types):
                raise TypeError(
                    'callable passed as {} has too many arguments in its declaration; expected {} '
                    'but {} argument(s) declared'.format(argname, len(argument_types),
                                                         num_mandatory_args))
            elif not has_varargs and num_mandatory_args < len(argument_types):
                raise TypeError(
                    'callable passed as {} has too few arguments in its declaration; expected {} '
                    'but {} argument(s) declared'.format(argname, len(argument_types),
                                                         num_mandatory_args))


def check_dict(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not isinstance(value, dict):
        raise TypeError('type of {} must be a dict; got {} instead'.
                        format(argname, qualified_name(value)))

    if expected_type is not dict:
        if (hasattr(expected_type, "__args__") and
                expected_type.__args__ not in (None, expected_type.__parameters__)):
            key_type, value_type = expected_type.__args__
            if key_type is not Any or value_type is not Any:
                for k, v in value.items():
                    check_type('keys of {}'.format(argname), k, key_type, memo)
                    check_type('{}[{!r}]'.format(argname, k), v, value_type, memo)


def check_typed_dict(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    declared_keys = frozenset(expected_type.__annotations__)
    if hasattr(expected_type, '__required_keys__'):
        required_keys = expected_type.__required_keys__
    else:  # py3.8 and lower
        required_keys = declared_keys if expected_type.__total__ else frozenset()

    existing_keys = frozenset(value)
    extra_keys = existing_keys - declared_keys
    if extra_keys:
        keys_formatted = ', '.join('"{}"'.format(key) for key in sorted(extra_keys))
        raise TypeError('extra key(s) ({}) in {}'.format(keys_formatted, argname))

    missing_keys = required_keys - existing_keys
    if missing_keys:
        keys_formatted = ', '.join('"{}"'.format(key) for key in sorted(missing_keys))
        raise TypeError('required key(s) ({}) missing from {}'.format(keys_formatted, argname))

    for key, argtype in get_type_hints(expected_type).items():
        argvalue = value.get(key, _missing)
        if argvalue is not _missing:
            check_type('dict item "{}" for {}'.format(key, argname), argvalue, argtype, memo)


def check_list(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not isinstance(value, list):
        raise TypeError('type of {} must be a list; got {} instead'.
                        format(argname, qualified_name(value)))

    if expected_type is not list:
        if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
                (None, expected_type.__parameters__):
            value_type = expected_type.__args__[0]
            if value_type is not Any:
                for i, v in enumerate(value):
                    check_type('{}[{}]'.format(argname, i), v, value_type, memo)


def check_sequence(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not isinstance(value, collections.abc.Sequence):
        raise TypeError('type of {} must be a sequence; got {} instead'.
                        format(argname, qualified_name(value)))

    if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
            (None, expected_type.__parameters__):
        value_type = expected_type.__args__[0]
        if value_type is not Any:
            for i, v in enumerate(value):
                check_type('{}[{}]'.format(argname, i), v, value_type, memo)


def check_set(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not isinstance(value, AbstractSet):
        raise TypeError('type of {} must be a set; got {} instead'.
                        format(argname, qualified_name(value)))

    if expected_type is not set:
        if hasattr(expected_type, "__args__") and expected_type.__args__ not in \
                (None, expected_type.__parameters__):
            value_type = expected_type.__args__[0]
            if value_type is not Any:
                for v in value:
                    check_type('elements of {}'.format(argname), v, value_type, memo)


def check_tuple(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    # Specialized check for NamedTuples
    is_named_tuple = False
    if sys.version_info < (3, 8, 0):
        is_named_tuple = hasattr(expected_type, '_field_types')  # deprecated since python 3.8
    else:
        is_named_tuple = hasattr(expected_type, '__annotations__')

    if is_named_tuple:
        if not isinstance(value, expected_type):
            raise TypeError('type of {} must be a named tuple of type {}; got {} instead'.
                            format(argname, qualified_name(expected_type), qualified_name(value)))

        if sys.version_info < (3, 8, 0):
            field_types = expected_type._field_types
        else:
            field_types = expected_type.__annotations__

        for name, field_type in field_types.items():
            check_type('{}.{}'.format(argname, name), getattr(value, name), field_type, memo)

        return
    elif not isinstance(value, tuple):
        raise TypeError('type of {} must be a tuple; got {} instead'.
                        format(argname, qualified_name(value)))

    if getattr(expected_type, '__tuple_params__', None):
        # Python 3.5
        use_ellipsis = expected_type.__tuple_use_ellipsis__
        tuple_params = expected_type.__tuple_params__
    elif getattr(expected_type, '__args__', None):
        # Python 3.6+
        use_ellipsis = expected_type.__args__[-1] is Ellipsis
        tuple_params = expected_type.__args__[:-1 if use_ellipsis else None]
    else:
        # Unparametrized Tuple or plain tuple
        return

    if use_ellipsis:
        element_type = tuple_params[0]
        for i, element in enumerate(value):
            check_type('{}[{}]'.format(argname, i), element, element_type, memo)
    elif tuple_params == ((),):
        if value != ():
            raise TypeError('{} is not an empty tuple but one was expected'.format(argname))
    else:
        if len(value) != len(tuple_params):
            raise TypeError('{} has wrong number of elements (expected {}, got {} instead)'
                            .format(argname, len(tuple_params), len(value)))

        for i, (element, element_type) in enumerate(zip(value, tuple_params)):
            check_type('{}[{}]'.format(argname, i), element, element_type, memo)


def check_union(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if hasattr(expected_type, '__union_params__'):
        # Python 3.5
        union_params = expected_type.__union_params__
    else:
        # Python 3.6+
        union_params = expected_type.__args__

    for type_ in union_params:
        try:
            check_type(argname, value, type_, memo)
            return
        except TypeError:
            pass

    typelist = ', '.join(get_type_name(t) for t in union_params)
    raise TypeError('type of {} must be one of ({}); got {} instead'.
                    format(argname, typelist, qualified_name(value)))


def check_class(argname: str, value, expected_type, memo: _TypeCheckMemo) -> None:
    if not isclass(value):
        raise TypeError('type of {} must be a type; got {} instead'.format(
            argname, qualified_name(value)))

    # Needed on Python 3.7+
    if expected_type is Type:
        return

    if getattr(expected_type, '__origin__', None) in (Type, type):
        expected_class = expected_type.__args__[0]
    else:
        expected_class = expected_type

    if expected_class is Any:
        return
    elif isinstance(expected_class, TypeVar):
        check_typevar(argname, value, expected_class, memo, True)
    elif getattr(expected_class, '__origin__', None) is Union:
        for arg in expected_class.__args__:
            try:
                check_class(argname, value, arg, memo)
                break
            except TypeError:
                pass
        else:
            formatted_args = ', '.join(get_type_name(arg) for arg in expected_class.__args__)
            raise TypeError('{} must match one of the following: ({}); got {} instead'.format(
                argname, formatted_args, qualified_name(value)
            ))
    elif not issubclass(value, expected_class):
        raise TypeError('{} must be a subclass of {}; got {} instead'.format(
            argname, qualified_name(expected_class), qualified_name(value)))


def check_typevar(argname: str, value, typevar: TypeVar, memo: _TypeCheckMemo,
                  subclass_check: bool = False) -> None:
    value_type = value if subclass_check else type(value)
    subject = argname if subclass_check else 'type of ' + argname

    if typevar.__bound__ is not None:
        bound_type = resolve_forwardref(typevar.__bound__, memo)
        if not issubclass(value_type, bound_type):
            raise TypeError(
                '{} must be {} or one of its subclasses; got {} instead'
                .format(subject, qualified_name(bound_type), qualified_name(value_type)))
    elif typevar.__constraints__:
        constraints = [resolve_forwardref(c, memo) for c in typevar.__constraints__]
        for constraint in constraints:
            try:
                check_type(argname, value, constraint, memo)
            except TypeError:
                pass
            else:
                break
        else:
            formatted_constraints = ', '.join(get_type_name(constraint)
                                              for constraint in constraints)
            raise TypeError('{} must match one of the constraints ({}); got {} instead'
                            .format(subject, formatted_constraints, qualified_name(value_type)))


def check_literal(argname: str, value, expected_type, memo: _TypeCheckMemo):
    def get_args(literal):
        try:
            args = literal.__args__
        except AttributeError:
            # Instance of Literal from typing_extensions
            args = literal.__values__

        retval = []
        for arg in args:
            if isinstance(arg, Literal.__class__) or getattr(arg, '__origin__', None) is Literal:
                # The first check works on py3.6 and lower, the second one on py3.7+
                retval.extend(get_args(arg))
            elif isinstance(arg, (int, str, bytes, bool, type(None), Enum)):
                retval.append(arg)
            else:
                raise TypeError('Illegal literal value: {}'.format(arg))

        return retval

    final_args = tuple(get_args(expected_type))
    if value not in final_args:
        raise TypeError('the value of {} must be one of {}; got {} instead'.
                        format(argname, final_args, value))


def check_number(argname: str, value, expected_type):
    if expected_type is complex and not isinstance(value, (complex, float, int)):
        raise TypeError('type of {} must be either complex, float or int; got {} instead'.
                        format(argname, qualified_name(value.__class__)))
    elif expected_type is float and not isinstance(value, (float, int)):
        raise TypeError('type of {} must be either float or int; got {} instead'.
                        format(argname, qualified_name(value.__class__)))


def check_io(argname: str, value, expected_type):
    if expected_type is TextIO:
        if not isinstance(value, TextIOBase):
            raise TypeError('type of {} must be a text based I/O object; got {} instead'.
                            format(argname, qualified_name(value.__class__)))
    elif expected_type is BinaryIO:
        if not isinstance(value, (RawIOBase, BufferedIOBase)):
            raise TypeError('type of {} must be a binary I/O object; got {} instead'.
                            format(argname, qualified_name(value.__class__)))
    elif not isinstance(value, IOBase):
        raise TypeError('type of {} must be an I/O object; got {} instead'.
                        format(argname, qualified_name(value.__class__)))


def check_protocol(argname: str, value, expected_type):
    # TODO: implement proper compatibility checking and support non-runtime protocols
    if getattr(expected_type, '_is_runtime_protocol', False):
        if not isinstance(value, expected_type):
            raise TypeError('type of {} ({}) is not compatible with the {} protocol'.
                            format(argname, type(value).__qualname__, expected_type.__qualname__))


# Equality checks are applied to these
origin_type_checkers = {
    AbstractSet: check_set,
    Callable: check_callable,
    collections.abc.Callable: check_callable,
    dict: check_dict,
    Dict: check_dict,
    list: check_list,
    List: check_list,
    Sequence: check_sequence,
    collections.abc.Sequence: check_sequence,
    collections.abc.Set: check_set,
    set: check_set,
    Set: check_set,
    tuple: check_tuple,
    Tuple: check_tuple,
    type: check_class,
    Type: check_class,
    Union: check_union
}
_subclass_check_unions = hasattr(Union, '__union_set_params__')
if Literal is not None:
    origin_type_checkers[Literal] = check_literal

generator_origin_types = (Generator, collections.abc.Generator,
                          Iterator, collections.abc.Iterator,
                          Iterable, collections.abc.Iterable)
asyncgen_origin_types = (AsyncIterator, collections.abc.AsyncIterator,
                         AsyncIterable, collections.abc.AsyncIterable)
if AsyncGenerator is not None:
    asyncgen_origin_types += (AsyncGenerator,)
if hasattr(collections.abc, 'AsyncGenerator'):
    asyncgen_origin_types += (collections.abc.AsyncGenerator,)


def check_type(argname: str, value, expected_type, memo: Optional[_TypeCheckMemo] = None, *,
               globals: Optional[Dict[str, Any]] = None,
               locals: Optional[Dict[str, Any]] = None) -> None:
    """
    Ensure that ``value`` matches ``expected_type``.

    The types from the :mod:`typing` module do not support :func:`isinstance` or :func:`issubclass`
    so a number of type specific checks are required. This function knows which checker to call
    for which type.

    :param argname: name of the argument to check; used for error messages
    :param value: value to be checked against ``expected_type``
    :param expected_type: a class or generic type instance
    :param globals: dictionary of global variables to use for resolving forward references
        (defaults to the calling frame's globals)
    :param locals: dictionary of local variables to use for resolving forward references
        (defaults to the calling frame's locals)
    :raises TypeError: if there is a type mismatch

    """
    if expected_type is Any or isinstance(value, Mock):
        return

    if expected_type is None:
        # Only happens on < 3.6
        expected_type = type(None)

    if memo is None:
        frame = sys._getframe(1)
        if globals is None:
            globals = frame.f_globals
        if locals is None:
            locals = frame.f_locals

        memo = _TypeCheckMemo(globals, locals)

    expected_type = resolve_forwardref(expected_type, memo)
    origin_type = getattr(expected_type, '__origin__', None)
    if origin_type is not None:
        checker_func = origin_type_checkers.get(origin_type)
        if checker_func:
            checker_func(argname, value, expected_type, memo)
        else:
            check_type(argname, value, origin_type, memo)
    elif isclass(expected_type):
        if issubclass(expected_type, Tuple):
            check_tuple(argname, value, expected_type, memo)
        elif issubclass(expected_type, (float, complex)):
            check_number(argname, value, expected_type)
        elif _subclass_check_unions and issubclass(expected_type, Union):
            check_union(argname, value, expected_type, memo)
        elif isinstance(expected_type, TypeVar):
            check_typevar(argname, value, expected_type, memo)
        elif issubclass(expected_type, IO):
            check_io(argname, value, expected_type)
        elif is_typeddict(expected_type):
            check_typed_dict(argname, value, expected_type, memo)
        elif getattr(expected_type, '_is_protocol', False):
            check_protocol(argname, value, expected_type)
        else:
            expected_type = (getattr(expected_type, '__extra__', None) or origin_type or
                             expected_type)

            if expected_type is bytes:
                # As per https://github.com/python/typing/issues/552
                if not isinstance(value, (bytearray, bytes, memoryview)):
                    raise TypeError('type of {} must be bytes-like; got {} instead'
                                    .format(argname, qualified_name(value)))
            elif not isinstance(value, expected_type):
                raise TypeError(
                    'type of {} must be {}; got {} instead'.
                    format(argname, qualified_name(expected_type), qualified_name(value)))
    elif isinstance(expected_type, TypeVar):
        # Only happens on < 3.6
        check_typevar(argname, value, expected_type, memo)
    elif isinstance(expected_type, Literal.__class__):
        # Only happens on < 3.7 when using Literal from typing_extensions
        check_literal(argname, value, expected_type, memo)
    elif expected_type.__class__ is NewType:
        # typing.NewType on Python 3.10+
        return check_type(argname, value, expected_type.__supertype__, memo)
    elif (isfunction(expected_type) and
            getattr(expected_type, "__module__", None) == "typing" and
            getattr(expected_type, "__qualname__", None).startswith("NewType.") and
            hasattr(expected_type, "__supertype__")):
        # typing.NewType on Python 3.9 and below
        return check_type(argname, value, expected_type.__supertype__, memo)


def check_return_type(retval, memo: Optional[_CallMemo] = None) -> bool:
    """
    Check that the return value is compatible with the return value annotation in the function.

    :param retval: the value about to be returned from the call
    :return: ``True``
    :raises TypeError: if there is a type mismatch

    """
    if memo is None:
        # faster than inspect.currentframe(), but not officially
        # supported in all python implementations
        frame = sys._getframe(1)

        try:
            func = find_function(frame)
        except LookupError:
            return True  # This can happen with the Pydev/PyCharm debugger extension installed

        memo = _CallMemo(func, frame.f_locals)

    if 'return' in memo.type_hints:
        if memo.type_hints['return'] is NoReturn:
            raise TypeError('{}() was declared never to return but it did'.format(memo.func_name))

        try:
            check_type('the return value', retval, memo.type_hints['return'], memo)
        except TypeError as exc:  # suppress unnecessarily long tracebacks
            # Allow NotImplemented if this is a binary magic method (__eq__() et al)
            if retval is NotImplemented and memo.type_hints['return'] is bool:
                # This does (and cannot) not check if it's actually a method
                func_name = memo.func_name.rsplit('.', 1)[-1]
                if len(memo.arguments) == 2 and func_name in BINARY_MAGIC_METHODS:
                    return True

            raise TypeError(*exc.args) from None

    return True


def check_argument_types(memo: Optional[_CallMemo] = None) -> bool:
    """
    Check that the argument values match the annotated types.

    Unless both ``args`` and ``kwargs`` are provided, the information will be retrieved from
    the previous stack frame (ie. from the function that called this).

    :return: ``True``
    :raises TypeError: if there is an argument type mismatch

    """
    if memo is None:
        # faster than inspect.currentframe(), but not officially
        # supported in all python implementations
        frame = sys._getframe(1)

        try:
            func = find_function(frame)
        except LookupError:
            return True  # This can happen with the Pydev/PyCharm debugger extension installed

        memo = _CallMemo(func, frame.f_locals)

    for argname, expected_type in memo.type_hints.items():
        if argname != 'return' and argname in memo.arguments:
            value = memo.arguments[argname]
            description = 'argument "{}"'.format(argname)
            try:
                check_type(description, value, expected_type, memo)
            except TypeError as exc:  # suppress unnecessarily long tracebacks
                raise TypeError(*exc.args) from None

    return True


class TypeCheckedGenerator:
    def __init__(self, wrapped: Generator, memo: _CallMemo):
        rtype_args = []
        if hasattr(memo.type_hints['return'], "__args__"):
            rtype_args = memo.type_hints['return'].__args__

        self.__wrapped = wrapped
        self.__memo = memo
        self.__yield_type = rtype_args[0] if rtype_args else Any
        self.__send_type = rtype_args[1] if len(rtype_args) > 1 else Any
        self.__return_type = rtype_args[2] if len(rtype_args) > 2 else Any
        self.__initialized = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped, name)

    def throw(self, *args):
        return self.__wrapped.throw(*args)

    def close(self):
        self.__wrapped.close()

    def send(self, obj):
        if self.__initialized:
            check_type('value sent to generator', obj, self.__send_type, memo=self.__memo)
        else:
            self.__initialized = True

        try:
            value = self.__wrapped.send(obj)
        except StopIteration as exc:
            check_type('return value', exc.value, self.__return_type, memo=self.__memo)
            raise

        check_type('value yielded from generator', value, self.__yield_type, memo=self.__memo)
        return value


class TypeCheckedAsyncGenerator:
    def __init__(self, wrapped: AsyncGenerator, memo: _CallMemo):
        rtype_args = memo.type_hints['return'].__args__
        self.__wrapped = wrapped
        self.__memo = memo
        self.__yield_type = rtype_args[0]
        self.__send_type = rtype_args[1] if len(rtype_args) > 1 else Any
        self.__initialized = False

    def __aiter__(self):
        return self

    def __anext__(self):
        return self.asend(None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped, name)

    def athrow(self, *args):
        return self.__wrapped.athrow(*args)

    def aclose(self):
        return self.__wrapped.aclose()

    async def asend(self, obj):
        if self.__initialized:
            check_type('value sent to generator', obj, self.__send_type, memo=self.__memo)
        else:
            self.__initialized = True

        value = await self.__wrapped.asend(obj)
        check_type('value yielded from generator', value, self.__yield_type, memo=self.__memo)
        return value


@overload
def typechecked(*, always: bool = False) -> Callable[[T_CallableOrType], T_CallableOrType]:
    ...


@overload
def typechecked(func: T_CallableOrType, *, always: bool = False) -> T_CallableOrType:
    ...


def typechecked(func=None, *, always=False, _localns: Optional[Dict[str, Any]] = None):
    """
    Perform runtime type checking on the arguments that are passed to the wrapped function.

    The return value is also checked against the return annotation if any.

    If the ``__debug__`` global variable is set to ``False``, no wrapping and therefore no type
    checking is done, unless ``always`` is ``True``.

    This can also be used as a class decorator. This will wrap all type annotated methods,
    including ``@classmethod``, ``@staticmethod``,  and ``@property`` decorated methods,
    in the class with the ``@typechecked`` decorator.

    :param func: the function or class to enable type checking for
    :param always: ``True`` to enable type checks even in optimized mode

    """
    if func is None:
        return partial(typechecked, always=always, _localns=_localns)

    if not __debug__ and not always:  # pragma: no cover
        return func

    if isclass(func):
        prefix = func.__qualname__ + '.'
        for key, attr in func.__dict__.items():
            if inspect.isfunction(attr) or inspect.ismethod(attr) or inspect.isclass(attr):
                if attr.__qualname__.startswith(prefix) and getattr(attr, '__annotations__', None):
                    setattr(func, key, typechecked(attr, always=always, _localns=func.__dict__))
            elif isinstance(attr, (classmethod, staticmethod)):
                if getattr(attr.__func__, '__annotations__', None):
                    wrapped = typechecked(attr.__func__, always=always, _localns=func.__dict__)
                    setattr(func, key, type(attr)(wrapped))
            elif isinstance(attr, property):
                kwargs = dict(doc=attr.__doc__)
                for name in ("fset", "fget", "fdel"):
                    property_func = kwargs[name] = getattr(attr, name)
                    if property_func is not None and getattr(property_func, '__annotations__', ()):
                        kwargs[name] = typechecked(
                            property_func, always=always, _localns=func.__dict__
                        )

                setattr(func, key, attr.__class__(**kwargs))

        return func

    if not getattr(func, '__annotations__', None):
        warn('no type annotations present -- not typechecking {}'.format(function_name(func)))
        return func

    # Find the frame in which the function was declared, for resolving forward references later
    if _localns is None:
        _localns = sys._getframe(1).f_locals

    # Find either the first Python wrapper or the actual function
    python_func = inspect.unwrap(func, stop=lambda f: hasattr(f, '__code__'))

    if not getattr(python_func, '__code__', None):
        warn('no code associated -- not typechecking {}'.format(function_name(func)))
        return func

    def wrapper(*args, **kwargs):
        memo = _CallMemo(python_func, _localns, args=args, kwargs=kwargs)
        check_argument_types(memo)
        retval = func(*args, **kwargs)
        try:
            check_return_type(retval, memo)
        except TypeError as exc:
            raise TypeError(*exc.args) from None

        # If a generator is returned, wrap it if its yield/send/return types can be checked
        if inspect.isgenerator(retval) or isasyncgen(retval):
            return_type = memo.type_hints.get('return')
            if return_type:
                origin = getattr(return_type, '__origin__', None)
                if origin in generator_origin_types:
                    return TypeCheckedGenerator(retval, memo)
                elif origin is not None and origin in asyncgen_origin_types:
                    return TypeCheckedAsyncGenerator(retval, memo)

        return retval

    async def async_wrapper(*args, **kwargs):
        memo = _CallMemo(python_func, _localns, args=args, kwargs=kwargs)
        check_argument_types(memo)
        retval = await func(*args, **kwargs)
        check_return_type(retval, memo)
        return retval

    if inspect.iscoroutinefunction(func):
        if python_func.__code__ is not async_wrapper.__code__:
            return wraps(func)(async_wrapper)
    else:
        if python_func.__code__ is not wrapper.__code__:
            return wraps(func)(wrapper)

    # the target callable was already wrapped
    return func


class TypeWarning(UserWarning):
    """
    A warning that is emitted when a type check fails.

    :ivar str event: ``call`` or ``return``
    :ivar Callable func: the function in which the violation occurred (the called function if event
        is ``call``, or the function where a value of the wrong type was returned from if event is
        ``return``)
    :ivar str error: the error message contained by the caught :class:`TypeError`
    :ivar frame: the frame in which the violation occurred
    """

    __slots__ = ('func', 'event', 'message', 'frame')

    def __init__(self, memo: Optional[_CallMemo], event: str, frame,
                 exception: Union[str, TypeError]):  # pragma: no cover
        self.func = memo.func
        self.event = event
        self.error = str(exception)
        self.frame = frame

        if self.event == 'call':
            caller_frame = self.frame.f_back
            event = 'call to {}() from {}:{}'.format(
                function_name(self.func), caller_frame.f_code.co_filename, caller_frame.f_lineno)
        else:
            event = 'return from {}() at {}:{}'.format(
                function_name(self.func), self.frame.f_code.co_filename, self.frame.f_lineno)

        super().__init__('[{thread_name}] {event}: {self.error}'.format(
            thread_name=threading.current_thread().name, event=event, self=self))

    @property
    def stack(self):
        """Return the stack where the last frame is from the target function."""
        return extract_stack(self.frame)

    def print_stack(self, file: TextIO = None, limit: int = None) -> None:
        """
        Print the traceback from the stack frame where the target function was run.

        :param file: an open file to print to (prints to stdout if omitted)
        :param limit: the maximum number of stack frames to print

        """
        print_stack(self.frame, limit, file)


class TypeChecker:
    """
    A type checker that collects type violations by hooking into :func:`sys.setprofile`.

    :param packages: list of top level modules and packages or modules to include for type checking
    :param all_threads: ``True`` to check types in all threads created while the checker is
        running, ``False`` to only check in the current one
    :param forward_refs_policy: how to handle unresolvable forward references in annotations

    .. deprecated:: 2.6
       Use :func:`~.importhook.install_import_hook` instead. This class will be removed in v3.0.
    """

    def __init__(self, packages: Union[str, Sequence[str]], *, all_threads: bool = True,
                 forward_refs_policy: ForwardRefPolicy = ForwardRefPolicy.ERROR):
        assert check_argument_types()
        warn('TypeChecker has been deprecated and will be removed in v3.0. '
             'Use install_import_hook() or the pytest plugin instead.', DeprecationWarning)
        self.all_threads = all_threads
        self.annotation_policy = forward_refs_policy
        self._call_memos = {}  # type: Dict[Any, _CallMemo]
        self._previous_profiler = None
        self._previous_thread_profiler = None
        self._active = False

        if isinstance(packages, str):
            self._packages = (packages,)
        else:
            self._packages = tuple(packages)

    @property
    def active(self) -> bool:
        """Return ``True`` if currently collecting type violations."""
        return self._active

    def should_check_type(self, func: Callable) -> bool:
        if not func.__annotations__:
            # No point in checking if there are no type hints
            return False
        elif isasyncgenfunction(func):
            # Async generators cannot be supported because the return arg is of an opaque builtin
            # type (async_generator_wrapped_value)
            return False
        else:
            # Check types if the module matches any of the package prefixes
            return any(func.__module__ == package or func.__module__.startswith(package + '.')
                       for package in self._packages)

    def start(self):
        if self._active:
            raise RuntimeError('type checker already running')

        self._active = True

        # Install this instance as the current profiler
        self._previous_profiler = sys.getprofile()
        sys.setprofile(self)

        # If requested, set this instance as the default profiler for all future threads
        # (does not affect existing threads)
        if self.all_threads:
            self._previous_thread_profiler = threading._profile_hook
            threading.setprofile(self)

    def stop(self):
        if self._active:
            if sys.getprofile() is self:
                sys.setprofile(self._previous_profiler)
            else:  # pragma: no cover
                warn('the system profiling hook has changed unexpectedly')

            if self.all_threads:
                if threading._profile_hook is self:
                    threading.setprofile(self._previous_thread_profiler)
                else:  # pragma: no cover
                    warn('the threading profiling hook has changed unexpectedly')

            self._active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __call__(self, frame, event: str, arg) -> None:  # pragma: no cover
        if not self._active:
            # This happens if all_threads was enabled and a thread was created when the checker was
            # running but was then stopped. The thread's profiler callback can't be reset any other
            # way but this.
            sys.setprofile(self._previous_thread_profiler)
            return

        # If an actual profiler is running, don't include the type checking times in its results
        if event == 'call':
            try:
                func = find_function(frame)
            except Exception:
                func = None

            if func is not None and self.should_check_type(func):
                memo = self._call_memos[frame] = _CallMemo(
                    func, frame.f_locals, forward_refs_policy=self.annotation_policy)
                if memo.is_generator:
                    return_type_hint = memo.type_hints['return']
                    if return_type_hint is not None:
                        origin = getattr(return_type_hint, '__origin__', None)
                        if origin in generator_origin_types:
                            # Check the types of the yielded values
                            memo.type_hints['return'] = return_type_hint.__args__[0]
                else:
                    try:
                        check_argument_types(memo)
                    except TypeError as exc:
                        warn(TypeWarning(memo, event, frame, exc))

            if self._previous_profiler is not None:
                self._previous_profiler(frame, event, arg)
        elif event == 'return':
            if self._previous_profiler is not None:
                self._previous_profiler(frame, event, arg)

            if arg is None:
                # a None return value might mean an exception is being raised but we have no way of
                # checking
                return

            memo = self._call_memos.get(frame)
            if memo is not None:
                try:
                    if memo.is_generator:
                        check_type('yielded value', arg, memo.type_hints['return'], memo)
                    else:
                        check_return_type(arg, memo)
                except TypeError as exc:
                    warn(TypeWarning(memo, event, frame, exc))

                if not memo.is_generator:
                    del self._call_memos[frame]
        elif self._previous_profiler is not None:
            self._previous_profiler(frame, event, arg)
