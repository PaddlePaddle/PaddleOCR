"""A module containing `numpy`-specific plugins for mypy."""

from __future__ import annotations

import typing as t

import numpy as np

try:
    import mypy.types
    from mypy.types import Type
    from mypy.plugin import Plugin, AnalyzeTypeContext
    from mypy.nodes import MypyFile, ImportFrom, Statement
    from mypy.build import PRI_MED

    _HookFunc = t.Callable[[AnalyzeTypeContext], Type]
    MYPY_EX: t.Optional[ModuleNotFoundError] = None
except ModuleNotFoundError as ex:
    MYPY_EX = ex

__all__: t.List[str] = []


def _get_precision_dict() -> t.Dict[str, str]:
    names = [
        ("_NBitByte", np.byte),
        ("_NBitShort", np.short),
        ("_NBitIntC", np.intc),
        ("_NBitIntP", np.intp),
        ("_NBitInt", np.int_),
        ("_NBitLongLong", np.longlong),

        ("_NBitHalf", np.half),
        ("_NBitSingle", np.single),
        ("_NBitDouble", np.double),
        ("_NBitLongDouble", np.longdouble),
    ]
    ret = {}
    for name, typ in names:
        n: int = 8 * typ().dtype.itemsize
        ret[f'numpy.typing._nbit.{name}'] = f"numpy._{n}Bit"
    return ret


def _get_extended_precision_list() -> t.List[str]:
    extended_types = [np.ulonglong, np.longlong, np.longdouble, np.clongdouble]
    extended_names = {
        "uint128",
        "uint256",
        "int128",
        "int256",
        "float80",
        "float96",
        "float128",
        "float256",
        "complex160",
        "complex192",
        "complex256",
        "complex512",
    }
    return [i.__name__ for i in extended_types if i.__name__ in extended_names]


#: A dictionary mapping type-aliases in `numpy.typing._nbit` to
#: concrete `numpy.typing.NBitBase` subclasses.
_PRECISION_DICT: t.Final = _get_precision_dict()

#: A list with the names of all extended precision `np.number` subclasses.
_EXTENDED_PRECISION_LIST: t.Final = _get_extended_precision_list()


def _hook(ctx: AnalyzeTypeContext) -> Type:
    """Replace a type-alias with a concrete ``NBitBase`` subclass."""
    typ, _, api = ctx
    name = typ.name.split(".")[-1]
    name_new = _PRECISION_DICT[f"numpy.typing._nbit.{name}"]
    return api.named_type(name_new)


if t.TYPE_CHECKING or MYPY_EX is None:
    def _index(iterable: t.Iterable[Statement], id: str) -> int:
        """Identify the first ``ImportFrom`` instance the specified `id`."""
        for i, value in enumerate(iterable):
            if getattr(value, "id", None) == id:
                return i
        else:
            raise ValueError("Failed to identify a `ImportFrom` instance "
                             f"with the following id: {id!r}")

    class _NumpyPlugin(Plugin):
        """A plugin for assigning platform-specific `numpy.number` precisions."""

        def get_type_analyze_hook(self, fullname: str) -> t.Optional[_HookFunc]:
            """Set the precision of platform-specific `numpy.number` subclasses.

            For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.
            """
            if fullname in _PRECISION_DICT:
                return _hook
            return None

        def get_additional_deps(self, file: MypyFile) -> t.List[t.Tuple[int, str, int]]:
            """Import platform-specific extended-precision `numpy.number` subclasses.

            For example: `numpy.float96`, `numpy.float128` and `numpy.complex256`.
            """
            ret = [(PRI_MED, file.fullname, -1)]
            if file.fullname == "numpy":
                # Import ONLY the extended precision types available to the
                # platform in question
                imports = ImportFrom(
                    "numpy.typing._extended_precision", 0,
                    names=[(v, v) for v in _EXTENDED_PRECISION_LIST],
                )
                imports.is_top_level = True

                # Replace the much broader extended-precision import
                # (defined in `numpy/__init__.pyi`) with a more specific one
                for lst in [file.defs, file.imports]:  # type: t.List[Statement]
                    i = _index(lst, "numpy.typing._extended_precision")
                    lst[i] = imports
            return ret

    def plugin(version: str) -> t.Type[_NumpyPlugin]:
        """An entry-point for mypy."""
        return _NumpyPlugin

else:
    def plugin(version: str) -> t.Type[_NumpyPlugin]:
        """An entry-point for mypy."""
        raise MYPY_EX
