from __future__ import annotations

from typing import (
    Callable,
    Hashable,
)
import warnings

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._typing import (
    Dtype,
    DtypeObj,
)
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_scalar,
    is_signed_integer_dtype,
    is_unsigned_integer_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)

_num_index_shared_docs = {}


_num_index_shared_docs[
    "class_descr"
] = """
    Immutable sequence used for indexing and alignment. The basic object
    storing axis labels for all pandas objects. %(klass)s is a special case
    of `Index` with purely %(ltype)s labels. %(extra)s.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype (default: %(dtype)s)
    copy : bool
        Make a copy of input ndarray.
    name : object
        Name to be stored in the index.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Index : The base pandas Index type.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
"""


class NumericIndex(Index):
    """
    Provide numeric type operations.

    This is an abstract class.
    """

    _index_descr_args = {
        "klass": "NumericIndex",
        "ltype": "integer or float",
        "dtype": "inferred",
        "extra": "",
    }
    _values: np.ndarray
    _default_dtype: np.dtype
    _dtype_validation_metadata: tuple[Callable[..., bool], str]

    _is_numeric_dtype = True
    _can_hold_strings = False

    @cache_readonly
    def _can_hold_na(self) -> bool:
        if is_float_dtype(self.dtype):
            return True
        else:
            return False

    _engine_types: dict[np.dtype, type[libindex.IndexEngine]] = {
        np.dtype(np.int8): libindex.Int8Engine,
        np.dtype(np.int16): libindex.Int16Engine,
        np.dtype(np.int32): libindex.Int32Engine,
        np.dtype(np.int64): libindex.Int64Engine,
        np.dtype(np.uint8): libindex.UInt8Engine,
        np.dtype(np.uint16): libindex.UInt16Engine,
        np.dtype(np.uint32): libindex.UInt32Engine,
        np.dtype(np.uint64): libindex.UInt64Engine,
        np.dtype(np.float32): libindex.Float32Engine,
        np.dtype(np.float64): libindex.Float64Engine,
    }

    @property
    def _engine_type(self):
        return self._engine_types[self.dtype]

    @cache_readonly
    def inferred_type(self) -> str:
        return {
            "i": "integer",
            "u": "integer",
            "f": "floating",
        }[self.dtype.kind]

    def __new__(cls, data=None, dtype: Dtype | None = None, copy=False, name=None):
        name = maybe_extract_name(name, data, cls)

        subarr = cls._ensure_array(data, dtype, copy)
        return cls._simple_new(subarr, name=name)

    @classmethod
    def _ensure_array(cls, data, dtype, copy: bool):
        """
        Ensure we have a valid array to pass to _simple_new.
        """
        cls._validate_dtype(dtype)

        if not isinstance(data, (np.ndarray, Index)):
            # Coerce to ndarray if not already ndarray or Index
            if is_scalar(data):
                raise cls._scalar_data_error(data)

            # other iterable of some kind
            if not isinstance(data, (ABCSeries, list, tuple)):
                data = list(data)

            data = np.asarray(data, dtype=dtype)

        if issubclass(data.dtype.type, str):
            cls._string_data_error(data)

        dtype = cls._ensure_dtype(dtype)

        if copy or not is_dtype_equal(data.dtype, dtype):
            subarr = np.array(data, dtype=dtype, copy=copy)
            cls._assert_safe_casting(data, subarr)
        else:
            subarr = data

        if subarr.ndim > 1:
            # GH#13601, GH#20285, GH#27125
            raise ValueError("Index data must be 1-dimensional")

        subarr = np.asarray(subarr)
        return subarr

    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None:
        if dtype is None:
            return

        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    @classmethod
    def _ensure_dtype(
        cls,
        dtype: Dtype | None,
    ) -> np.dtype | None:
        """Ensure int64 dtype for Int64Index, etc. Assumed dtype is validated."""
        return cls._default_dtype

    def __contains__(self, key) -> bool:
        """
        Check if key is a float and has a decimal. If it has, return False.
        """
        if not is_integer_dtype(self.dtype):
            return super().__contains__(key)

        hash(key)
        try:
            if is_float(key) and int(key) != key:
                # otherwise the `key in self._engine` check casts e.g. 1.1 -> 1
                return False
            return key in self._engine
        except (OverflowError, TypeError, ValueError):
            return False

    @doc(Index.astype)
    def astype(self, dtype, copy=True):
        if is_float_dtype(self.dtype):
            dtype = pandas_dtype(dtype)
            if needs_i8_conversion(dtype):
                raise TypeError(
                    f"Cannot convert Float64Index to dtype {dtype}; integer "
                    "values are required for conversion"
                )
            elif is_integer_dtype(dtype) and not is_extension_array_dtype(dtype):
                # TODO(jreback); this can change once we have an EA Index type
                # GH 13149
                arr = astype_nansafe(self._values, dtype=dtype)
                return Int64Index(arr, name=self.name)

        return super().astype(dtype, copy=copy)

    # ----------------------------------------------------------------
    # Indexing Methods

    @doc(Index._should_fallback_to_positional)
    def _should_fallback_to_positional(self) -> bool:
        return False

    @doc(Index._convert_slice_indexer)
    def _convert_slice_indexer(self, key: slice, kind: str):
        if is_float_dtype(self.dtype):
            assert kind in ["loc", "getitem"]

            # We always treat __getitem__ slicing as label-based
            # translate to locations
            return self.slice_indexer(key.start, key.stop, key.step, kind=kind)

        return super()._convert_slice_indexer(key, kind=kind)

    @doc(Index._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str, kind=lib.no_default):
        assert kind in ["loc", "getitem", None, lib.no_default]
        self._deprecated_arg(kind, "kind", "_maybe_cast_slice_bound")

        # we will try to coerce to integers
        return self._maybe_cast_indexer(label)

    # ----------------------------------------------------------------

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values, name: Hashable = lib.no_default):
        if not self._can_hold_na and values.dtype.kind == "f":
            name = self._name if name is lib.no_default else name
            # Ensure we are not returning an Int64Index with float data:
            return Float64Index._simple_new(values, name=name)
        return super()._shallow_copy(values=values, name=name)

    def _convert_tolerance(self, tolerance, target):
        tolerance = super()._convert_tolerance(tolerance, target)

        if not np.issubdtype(tolerance.dtype, np.number):
            if tolerance.ndim > 0:
                raise ValueError(
                    f"tolerance argument for {type(self).__name__} must contain "
                    "numeric elements if it is list type"
                )
            else:
                raise ValueError(
                    f"tolerance argument for {type(self).__name__} must be numeric "
                    f"if it is a scalar: {repr(tolerance)}"
                )
        return tolerance

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # If we ever have BoolIndex or ComplexIndex, this may need to be tightened
        return is_numeric_dtype(dtype)

    @classmethod
    def _assert_safe_casting(cls, data: np.ndarray, subarr: np.ndarray) -> None:
        """
        Ensure incoming data can be represented with matching signed-ness.

        Needed if the process of casting data from some accepted dtype to the internal
        dtype(s) bears the risk of truncation (e.g. float to int).
        """
        if is_integer_dtype(subarr.dtype):
            if not np.array_equal(data, subarr):
                raise TypeError("Unsafe NumPy casting, you must explicitly cast")

    @property
    def _is_all_dates(self) -> bool:
        """
        Checks that all the labels are datetime objects.
        """
        return False

    def _format_native_types(
        self, na_rep="", float_format=None, decimal=".", quoting=None, **kwargs
    ):
        from pandas.io.formats.format import FloatArrayFormatter

        if is_float_dtype(self.dtype):
            formatter = FloatArrayFormatter(
                self._values,
                na_rep=na_rep,
                float_format=float_format,
                decimal=decimal,
                quoting=quoting,
                fixed_width=False,
            )
            return formatter.get_result_as_array()

        return super()._format_native_types(
            na_rep=na_rep,
            float_format=float_format,
            decimal=decimal,
            quoting=quoting,
            **kwargs,
        )


class IntegerIndex(NumericIndex):
    """
    This is an abstract class for Int64Index, UInt64Index.
    """

    @property
    def asi8(self) -> np.ndarray:
        # do not cache or you'll create a memory leak
        warnings.warn(
            "Index.asi8 is deprecated and will be removed in a future version",
            FutureWarning,
            stacklevel=2,
        )
        return self._values.view(self._default_dtype)


class Int64Index(IntegerIndex):
    _index_descr_args = {
        "klass": "Int64Index",
        "ltype": "integer",
        "dtype": "int64",
        "extra": "",
    }
    __doc__ = _num_index_shared_docs["class_descr"] % _index_descr_args

    _typ = "int64index"
    _engine_type = libindex.Int64Engine
    _default_dtype = np.dtype(np.int64)
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")


class UInt64Index(IntegerIndex):
    _index_descr_args = {
        "klass": "UInt64Index",
        "ltype": "unsigned integer",
        "dtype": "uint64",
        "extra": "",
    }
    __doc__ = _num_index_shared_docs["class_descr"] % _index_descr_args

    _typ = "uint64index"
    _engine_type = libindex.UInt64Engine
    _default_dtype = np.dtype(np.uint64)
    _dtype_validation_metadata = (is_unsigned_integer_dtype, "unsigned integer")

    def _validate_fill_value(self, value):
        # e.g. np.array([1]) we want np.array([1], dtype=np.uint64)
        #  see test_where_uin64
        super()._validate_fill_value(value)
        if hasattr(value, "dtype") and is_signed_integer_dtype(value.dtype):
            if (value >= 0).all():
                return value.astype(self.dtype)
            raise TypeError
        return value


class Float64Index(NumericIndex):
    _index_descr_args = {
        "klass": "Float64Index",
        "dtype": "float64",
        "ltype": "float",
        "extra": "",
    }
    __doc__ = _num_index_shared_docs["class_descr"] % _index_descr_args

    _typ = "float64index"
    _engine_type = libindex.Float64Engine
    _default_dtype = np.dtype(np.float64)
    _dtype_validation_metadata = (is_float_dtype, "float")
