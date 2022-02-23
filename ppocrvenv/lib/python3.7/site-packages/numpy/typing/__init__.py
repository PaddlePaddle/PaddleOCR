"""
============================
Typing (:mod:`numpy.typing`)
============================

.. warning::

  Some of the types in this module rely on features only present in
  the standard library in Python 3.8 and greater. If you want to use
  these types in earlier versions of Python, you should install the
  typing-extensions_ package.

Large parts of the NumPy API have PEP-484-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Mypy plugin
-----------

A mypy_ plugin is distributed in `numpy.typing` for managing a number of
platform-specific annotations. Its function can be split into to parts:

* Assigning the (platform-dependent) precisions of certain `~numpy.number` subclasses,
  including the likes of `~numpy.int_`, `~numpy.intp` and `~numpy.longlong`.
  See the documentation on :ref:`scalar types <arrays.scalars.built-in>` for a
  comprehensive overview of the affected classes. without the plugin the precision
  of all relevant classes will be inferred as `~typing.Any`.
* Removing all extended-precision `~numpy.number` subclasses that are unavailable
  for the platform in question. Most notable this includes the likes of
  `~numpy.float128` and `~numpy.complex256`. Without the plugin *all*
  extended-precision types will, as far as mypy is concerned, be available
  to all platforms.

To enable the plugin, one must add it to their mypy `configuration file`_:

.. code-block:: ini

    [mypy]
    plugins = numpy.typing.mypy_plugin

.. _mypy: http://mypy-lang.org/
.. _configuration file: https://mypy.readthedocs.io/en/stable/config_file.html

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool_

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should instead use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as a covariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: "np.floating[T]", b: "np.floating[T]") -> "np.floating[T]":
    ...     ...

Consequently, the likes of `~numpy.float16`, `~numpy.float32` and
`~numpy.float64` are still sub-types of `~numpy.floating`, but, contrary to
runtime, they're not necessarily considered as sub-classes.

Timedelta64
~~~~~~~~~~~

The `~numpy.timedelta64` class is not considered a subclass of `~numpy.signedinteger`,
the former only inheriting from `~numpy.generic` while static type checking.

0D arrays
~~~~~~~~~

During runtime numpy aggressively casts any passed 0D arrays into their
corresponding `~numpy.generic` instance. Until the introduction of shape
typing (see :pep:`646`) it is unfortunately not possible to make the
necessary distinction between 0D and >0D arrays. While thus not strictly
correct, all operations are that can potentially perform a 0D-array -> scalar
cast are currently annotated as exclusively returning an `ndarray`.

If it is known in advance that an operation _will_ perform a
0D-array -> scalar cast, then one can consider manually remedying the
situation with either `typing.cast` or a ``# type: ignore`` comment.

API
---

"""
# NOTE: The API section will be appended with additional entries
# further down in this file

from numpy import ufunc
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 8):
        from typing import final
    else:
        from typing_extensions import final
else:
    def final(f): return f

if not TYPE_CHECKING:
    __all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]
else:
    # Ensure that all objects within this module are accessible while
    # static type checking. This includes private ones, as we need them
    # for internal use.
    #
    # Declare to mypy that `__all__` is a list of strings without assigning
    # an explicit value
    __all__: List[str]


@final  # Dissallow the creation of arbitrary `NBitBase` subclasses
class NBitBase:
    """
    An object representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating a
    function that takes a float and integer of arbitrary precision as arguments
    and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, Union, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T1 = TypeVar("T1", bound=npt.NBitBase)
        >>> T2 = TypeVar("T2", bound=npt.NBitBase)

        >>> def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[Union[T1, T2]]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()


# Silence errors about subclassing a `@final`-decorated class
class _256Bit(NBitBase): ...  # type: ignore[misc]
class _128Bit(_256Bit): ...  # type: ignore[misc]
class _96Bit(_128Bit): ...  # type: ignore[misc]
class _80Bit(_96Bit): ...  # type: ignore[misc]
class _64Bit(_80Bit): ...  # type: ignore[misc]
class _32Bit(_64Bit): ...  # type: ignore[misc]
class _16Bit(_32Bit): ...  # type: ignore[misc]
class _8Bit(_16Bit): ...  # type: ignore[misc]

from ._nbit import (
    _NBitByte,
    _NBitShort,
    _NBitIntC,
    _NBitIntP,
    _NBitInt,
    _NBitLongLong,
    _NBitHalf,
    _NBitSingle,
    _NBitDouble,
    _NBitLongDouble,
)
from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)
from ._scalars import (
    _CharLike_co,
    _BoolLike_co,
    _UIntLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _TD64Like_co,
    _NumberLike_co,
    _ScalarLike_co,
    _VoidLike_co,
)
from ._shape import _Shape, _ShapeLike
from ._dtype_like import (
    DTypeLike as DTypeLike,
    _SupportsDType,
    _VoidDTypeLike,
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
    _DTypeLikeTD64,
    _DTypeLikeDT64,
    _DTypeLikeObject,
    _DTypeLikeVoid,
    _DTypeLikeStr,
    _DTypeLikeBytes,
    _DTypeLikeComplex_co,
)
from ._array_like import (
    ArrayLike as ArrayLike,
    _ArrayLike,
    _NestedSequence,
    _RecursiveSequence,
    _SupportsArray,
    _ArrayLikeInt,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeVoid_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,
)
from ._generic_alias import (
    NDArray as NDArray,
    _GenericAlias,
)

if TYPE_CHECKING:
    from ._ufunc import (
        _UFunc_Nin1_Nout1,
        _UFunc_Nin2_Nout1,
        _UFunc_Nin1_Nout2,
        _UFunc_Nin2_Nout2,
        _GUFunc_Nin2_Nout1,
    )
else:
    # Declare the (type-check-only) ufunc subclasses as ufunc aliases during
    # runtime; this helps autocompletion tools such as Jedi (numpy/numpy#19834)
    _UFunc_Nin1_Nout1 = ufunc
    _UFunc_Nin2_Nout1 = ufunc
    _UFunc_Nin1_Nout2 = ufunc
    _UFunc_Nin2_Nout2 = ufunc
    _GUFunc_Nin2_Nout1 = ufunc

# Clean up the namespace
del TYPE_CHECKING, final, List, ufunc

if __doc__ is not None:
    from ._add_docstring import _docstrings
    __doc__ += _docstrings
    __doc__ += '\n.. autoclass:: numpy.typing.NBitBase\n'
    del _docstrings

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
