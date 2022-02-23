from __future__ import annotations

import sys
import types
from typing import (
    Any,
    ClassVar,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Tuple,
    Type,
    TypeVar,
    TYPE_CHECKING,
)

import numpy as np

__all__ = ["_GenericAlias", "NDArray"]

_T = TypeVar("_T", bound="_GenericAlias")


def _to_str(obj: object) -> str:
    """Helper function for `_GenericAlias.__repr__`."""
    if obj is Ellipsis:
        return '...'
    elif isinstance(obj, type) and not isinstance(obj, _GENERIC_ALIAS_TYPE):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        else:
            return f'{obj.__module__}.{obj.__qualname__}'
    else:
        return repr(obj)


def _parse_parameters(args: Iterable[Any]) -> Generator[TypeVar, None, None]:
    """Search for all typevars and typevar-containing objects in `args`.

    Helper function for `_GenericAlias.__init__`.

    """
    for i in args:
        if hasattr(i, "__parameters__"):
            yield from i.__parameters__
        elif isinstance(i, TypeVar):
            yield i


def _reconstruct_alias(alias: _T, parameters: Iterator[TypeVar]) -> _T:
    """Recursivelly replace all typevars with those from `parameters`.

    Helper function for `_GenericAlias.__getitem__`.

    """
    args = []
    for i in alias.__args__:
        if isinstance(i, TypeVar):
            value: Any = next(parameters)
        elif isinstance(i, _GenericAlias):
            value = _reconstruct_alias(i, parameters)
        elif hasattr(i, "__parameters__"):
            prm_tup = tuple(next(parameters) for _ in i.__parameters__)
            value = i[prm_tup]
        else:
            value = i
        args.append(value)

    cls = type(alias)
    return cls(alias.__origin__, tuple(args))


class _GenericAlias:
    """A python-based backport of the `types.GenericAlias` class.

    E.g. for ``t = list[int]``, ``t.__origin__`` is ``list`` and
    ``t.__args__`` is ``(int,)``.

    See Also
    --------
    :pep:`585`
        The PEP responsible for introducing `types.GenericAlias`.

    """

    __slots__ = ("__weakref__", "_origin", "_args", "_parameters", "_hash")

    @property
    def __origin__(self) -> type:
        return super().__getattribute__("_origin")

    @property
    def __args__(self) -> Tuple[Any, ...]:
        return super().__getattribute__("_args")

    @property
    def __parameters__(self) -> Tuple[TypeVar, ...]:
        """Type variables in the ``GenericAlias``."""
        return super().__getattribute__("_parameters")

    def __init__(self, origin: type, args: Any) -> None:
        self._origin = origin
        self._args = args if isinstance(args, tuple) else (args,)
        self._parameters = tuple(_parse_parameters(args))

    @property
    def __call__(self) -> type:
        return self.__origin__

    def __reduce__(self: _T) -> Tuple[Type[_T], Tuple[type, Tuple[Any, ...]]]:
        cls = type(self)
        return cls, (self.__origin__, self.__args__)

    def __mro_entries__(self, bases: Iterable[object]) -> Tuple[type]:
        return (self.__origin__,)

    def __dir__(self) -> List[str]:
        """Implement ``dir(self)``."""
        cls = type(self)
        dir_origin = set(dir(self.__origin__))
        return sorted(cls._ATTR_EXCEPTIONS | dir_origin)

    def __hash__(self) -> int:
        """Return ``hash(self)``."""
        # Attempt to use the cached hash
        try:
            return super().__getattribute__("_hash")
        except AttributeError:
            self._hash: int = hash(self.__origin__) ^ hash(self.__args__)
            return super().__getattribute__("_hash")

    def __instancecheck__(self, obj: object) -> NoReturn:
        """Check if an `obj` is an instance."""
        raise TypeError("isinstance() argument 2 cannot be a "
                        "parameterized generic")

    def __subclasscheck__(self, cls: type) -> NoReturn:
        """Check if a `cls` is a subclass."""
        raise TypeError("issubclass() argument 2 cannot be a "
                        "parameterized generic")

    def __repr__(self) -> str:
        """Return ``repr(self)``."""
        args = ", ".join(_to_str(i) for i in self.__args__)
        origin = _to_str(self.__origin__)
        return f"{origin}[{args}]"

    def __getitem__(self: _T, key: Any) -> _T:
        """Return ``self[key]``."""
        key_tup = key if isinstance(key, tuple) else (key,)

        if len(self.__parameters__) == 0:
            raise TypeError(f"There are no type variables left in {self}")
        elif len(key_tup) > len(self.__parameters__):
            raise TypeError(f"Too many arguments for {self}")
        elif len(key_tup) < len(self.__parameters__):
            raise TypeError(f"Too few arguments for {self}")

        key_iter = iter(key_tup)
        return _reconstruct_alias(self, key_iter)

    def __eq__(self, value: object) -> bool:
        """Return ``self == value``."""
        if not isinstance(value, _GENERIC_ALIAS_TYPE):
            return NotImplemented
        return (
            self.__origin__ == value.__origin__ and
            self.__args__ == value.__args__
        )

    _ATTR_EXCEPTIONS: ClassVar[FrozenSet[str]] = frozenset({
        "__origin__",
        "__args__",
        "__parameters__",
        "__mro_entries__",
        "__reduce__",
        "__reduce_ex__",
        "__copy__",
        "__deepcopy__",
    })

    def __getattribute__(self, name: str) -> Any:
        """Return ``getattr(self, name)``."""
        # Pull the attribute from `__origin__` unless its
        # name is in `_ATTR_EXCEPTIONS`
        cls = type(self)
        if name in cls._ATTR_EXCEPTIONS:
            return super().__getattribute__(name)
        return getattr(self.__origin__, name)


# See `_GenericAlias.__eq__`
if sys.version_info >= (3, 9):
    _GENERIC_ALIAS_TYPE = (_GenericAlias, types.GenericAlias)
else:
    _GENERIC_ALIAS_TYPE = (_GenericAlias,)

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)

if TYPE_CHECKING:
    NDArray = np.ndarray[Any, np.dtype[ScalarType]]
elif sys.version_info >= (3, 9):
    _DType = types.GenericAlias(np.dtype, (ScalarType,))
    NDArray = types.GenericAlias(np.ndarray, (Any, _DType))
else:
    _DType = _GenericAlias(np.dtype, (ScalarType,))
    NDArray = _GenericAlias(np.ndarray, (Any, _DType))
