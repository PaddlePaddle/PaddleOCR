from __future__ import annotations

from typing import (
    Any,
    Hashable,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import index as libindex
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
)
from pandas.util._decorators import (
    Appender,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_categorical_dtype,
    is_scalar,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core import accessor
from pandas.core.arrays.categorical import (
    Categorical,
    contains,
)
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    maybe_extract_name,
)
from pandas.core.indexes.extension import (
    NDArrayBackedExtensionIndex,
    inherit_names,
)

_index_doc_kwargs: dict[str, str] = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "CategoricalIndex"})


@inherit_names(
    [
        "argsort",
        "_internal_get_values",
        "tolist",
        "codes",
        "categories",
        "ordered",
        "_reverse_indexer",
        "searchsorted",
        "is_dtype_equal",
        "min",
        "max",
    ],
    Categorical,
)
@accessor.delegate_names(
    delegate=Categorical,
    accessors=[
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    typ="method",
    overwrite=True,
)
class CategoricalIndex(NDArrayBackedExtensionIndex, accessor.PandasDelegate):
    """
    Index based on an underlying :class:`Categorical`.

    CategoricalIndex, like Categorical, can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    like Categorical, it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    codes
    categories
    ordered

    Methods
    -------
    rename_categories
    reorder_categories
    add_categories
    remove_categories
    remove_unused_categories
    set_categories
    as_ordered
    as_unordered
    map

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    Index : The base pandas Index type.
    Categorical : A categorical array.
    CategoricalDtype : Type for categorical data.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#categoricalindex>`__
    for more.

    Examples
    --------
    >>> pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:

    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])
    >>> pd.CategoricalIndex(c)
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    Ordered ``CategoricalIndex`` can have a min and max value.

    >>> ci = pd.CategoricalIndex(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> ci
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['c', 'b', 'a'], ordered=True, dtype='category')
    >>> ci.min()
    'c'
    """

    _typ = "categoricalindex"
    _data_cls = Categorical

    @property
    def _can_hold_strings(self):
        return self.categories._can_hold_strings

    codes: np.ndarray
    categories: Index
    _data: Categorical
    _values: Categorical

    @property
    def _engine_type(self):
        # self.codes can have dtype int8, int16, int32 or int64, so we need
        # to return the corresponding engine type (libindex.Int8Engine, etc.).
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine,
        }[self.codes.dtype.type]

    _attributes = ["name"]

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable = None,
    ) -> CategoricalIndex:

        name = maybe_extract_name(name, data, cls)

        if data is None:
            # GH#38944
            warnings.warn(
                "Constructing a CategoricalIndex without passing data is "
                "deprecated and will raise in a future version. "
                "Use CategoricalIndex([], ...) instead",
                FutureWarning,
                stacklevel=2,
            )
            data = []

        if is_scalar(data):
            raise cls._scalar_data_error(data)

        data = Categorical(
            data, categories=categories, ordered=ordered, dtype=dtype, copy=copy
        )

        return cls._simple_new(data, name=name)

    # --------------------------------------------------------------------

    def _is_dtype_compat(self, other) -> Categorical:
        """
        *this is an internal non-public method*

        provide a comparison between the dtype of self and other (coercing if
        needed)

        Parameters
        ----------
        other : Index

        Returns
        -------
        Categorical

        Raises
        ------
        TypeError if the dtypes are not compatible
        """
        if is_categorical_dtype(other):
            other = extract_array(other)
            if not other._categories_match_up_to_permutation(self):
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        elif other._is_multi:
            # preempt raising NotImplementedError in isna call
            raise TypeError("MultiIndex is not dtype-compatible with CategoricalIndex")
        else:
            values = other

            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if not other.isin(values).all():
                raise TypeError(
                    "cannot append a non-category item to a CategoricalIndex"
                )
            other = other._values

            if not ((other == values) | (isna(other) & isna(values))).all():
                # GH#37667 see test_equals_non_category
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        return other

    def equals(self, other: object) -> bool:
        """
        Determine if two CategoricalIndex objects contain the same elements.

        Returns
        -------
        bool
            If two CategoricalIndex objects have equal elements True,
            otherwise False.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False

        return self._data.equals(other)

    # --------------------------------------------------------------------
    # Rendering Methods

    @property
    def _formatter_func(self):
        return self.categories._formatter_func

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        max_categories = (
            10
            if get_option("display.max_categories") == 0
            else get_option("display.max_categories")
        )
        attrs = [
            (
                "categories",
                ibase.default_pprint(self.categories, max_seq_items=max_categories),
            ),
            # error: "CategoricalIndex" has no attribute "ordered"
            ("ordered", self.ordered),  # type: ignore[attr-defined]
        ]
        extra = super()._format_attrs()
        return attrs + extra

    def _format_with_header(self, header: list[str], na_rep: str = "NaN") -> list[str]:
        from pandas.io.formats.printing import pprint_thing

        result = [
            pprint_thing(x, escape_chars=("\t", "\r", "\n")) if notna(x) else na_rep
            for x in self._values
        ]
        return header + result

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        return "categorical"

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        # if key is a NaN, check if any NaN is in self.
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans

        return contains(self, key, container=self._engine)

    @doc(Index.fillna)
    def fillna(self, value, downcast=None):
        value = self._require_scalar(value)
        try:
            cat = self._data.fillna(value)
        except (ValueError, TypeError):
            # invalid fill_value
            if not self.isna().any():
                # nothing to fill, we can get away without casting
                return self.copy()
            return self.astype(object).fillna(value, downcast=downcast)

        return type(self)._simple_new(cat, name=self.name)

    def reindex(
        self, target, method=None, level=None, limit=None, tolerance=None
    ) -> tuple[Index, np.ndarray | None]:
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index

        """
        if method is not None:
            raise NotImplementedError(
                "argument method is not implemented for CategoricalIndex.reindex"
            )
        if level is not None:
            raise NotImplementedError(
                "argument level is not implemented for CategoricalIndex.reindex"
            )
        if limit is not None:
            raise NotImplementedError(
                "argument limit is not implemented for CategoricalIndex.reindex"
            )

        target = ibase.ensure_index(target)

        if self.equals(target):
            indexer = None
            missing = np.array([], dtype=np.intp)
        else:
            indexer, missing = self.get_indexer_non_unique(np.array(target))

        if len(self.codes) and indexer is not None:
            new_target = self.take(indexer)
        else:
            new_target = target

        # filling in missing if needed
        if len(missing):
            cats = self.categories.get_indexer(target)

            if not isinstance(cats, CategoricalIndex) or (cats == -1).any():
                # coerce to a regular index here!
                result = Index(np.array(self), name=self.name)
                new_target, indexer, _ = result._reindex_non_unique(target)
            else:

                codes = new_target.codes.copy()
                codes[indexer == -1] = cats[missing]
                cat = self._data._from_backing_data(codes)
                new_target = type(self)._simple_new(cat, name=self.name)

        # we always want to return an Index type here
        # to be consistent with .reindex for other index types (e.g. they don't
        # coerce based on the actual values, only on the dtype)
        # unless we had an initial Categorical to begin with
        # in which case we are going to conform to the passed Categorical
        new_target = np.asarray(new_target)
        if is_categorical_dtype(target):
            cat = Categorical(new_target, dtype=target.dtype)
            new_target = type(self)._simple_new(cat, name=self.name)
        else:
            new_target = Index(new_target, name=self.name)

        return new_target, indexer

    # error: Return type "Tuple[Index, Optional[ndarray], Optional[ndarray]]"
    # of "_reindex_non_unique" incompatible with return type
    # "Tuple[Index, ndarray, Optional[ndarray]]" in supertype "Index"
    def _reindex_non_unique(  # type: ignore[override]
        self, target: Index
    ) -> tuple[Index, np.ndarray | None, np.ndarray | None]:
        """
        reindex from a non-unique; which CategoricalIndex's are almost
        always
        """
        # TODO: rule out `indexer is None` here to make the signature
        #  match the parent class's signature. This should be equivalent
        #  to ruling out `self.equals(target)`
        new_target, indexer = self.reindex(target)
        new_indexer = None

        check = indexer == -1
        # error: Item "bool" of "Union[Any, bool]" has no attribute "any"
        if check.any():  # type: ignore[union-attr]
            new_indexer = np.arange(len(self.take(indexer)), dtype=np.intp)
            new_indexer[check] = -1

        cats = self.categories.get_indexer(target)
        if not (cats == -1).any():
            # .reindex returns normal Index. Revert to CategoricalIndex if
            # all targets are included in my categories
            cat = Categorical(new_target, dtype=self.dtype)
            new_target = type(self)._simple_new(cat, name=self.name)

        return new_target, indexer, new_indexer

    # --------------------------------------------------------------------
    # Indexing Methods

    def _maybe_cast_indexer(self, key) -> int:
        # GH#41933: we have to do this instead of self._data._validate_scalar
        #  because this will correctly get partial-indexing on Interval categories
        try:
            return self._data._unbox_scalar(key)
        except KeyError:
            if is_valid_na_for_dtype(key, self.categories.dtype):
                return -1
            raise

    def _maybe_cast_listlike_indexer(self, values) -> CategoricalIndex:
        if isinstance(values, CategoricalIndex):
            values = values._data
        if isinstance(values, Categorical):
            # Indexing on codes is more efficient if categories are the same,
            #  so we can apply some optimizations based on the degree of
            #  dtype-matching.
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)
            codes = codes.astype(self.codes.dtype, copy=False)
            cat = self._data._from_backing_data(codes)
        return type(self)._simple_new(cat)

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> np.ndarray:
        # returned ndarray is np.intp

        if self.equals(target):
            return np.arange(len(self), dtype="intp")

        return self._get_indexer_non_unique(target._values)[0]

    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target) -> tuple[np.ndarray, np.ndarray]:
        # both returned ndarrays are np.intp
        target = ibase.ensure_index(target)
        return self._get_indexer_non_unique(target._values)

    def _get_indexer_non_unique(
        self, values: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        # both returned ndarrays are np.intp
        """
        get_indexer_non_unique but after unrapping the target Index object.
        """
        # Note: we use engine.get_indexer_non_unique for get_indexer in addition
        #  to get_indexer_non_unique because, even if `target` is unique, any
        #  non-category entries in it will be encoded as -1  so `codes` may
        #  not be unique.

        if isinstance(values, Categorical):
            # Indexing on codes is more efficient if categories are the same,
            #  so we can apply some optimizations based on the degree of
            #  dtype-matching.
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)

        indexer, missing = self._engine.get_indexer_non_unique(codes)
        return ensure_platform_int(indexer), ensure_platform_int(missing)

    # --------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return self.categories._is_comparable_dtype(dtype)

    def take_nd(self, *args, **kwargs):
        """Alias for `take`"""
        warnings.warn(
            "CategoricalIndex.take_nd is deprecated, use CategoricalIndex.take instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.take(*args, **kwargs)

    def map(self, mapper):
        """
        Map values using input correspondence (a dict, Series, or function).

        Maps the values (their categories, not the codes) of the index to new
        categories. If the mapping correspondence is one-to-one the result is a
        :class:`~pandas.CategoricalIndex` which has the same order property as
        the original, otherwise an :class:`~pandas.Index` is returned.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.

        Returns
        -------
        pandas.CategoricalIndex or pandas.Index
            Mapped index.

        See Also
        --------
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'])
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                          ordered=False, dtype='category')
        >>> idx.map(lambda x: x.upper())
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'third'})
        CategoricalIndex(['first', 'second', 'third'], categories=['first',
                         'second', 'third'], ordered=False, dtype='category')

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'], ordered=True)
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> idx.map({'a': 3, 'b': 2, 'c': 1})
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,
                         dtype='category')

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'first'})
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> idx.map({'a': 'first', 'b': 'second'})
        Index(['first', 'second', nan], dtype='object')
        """
        mapped = self._values.map(mapper)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat: list[Index], name: Hashable) -> Index:
        # if calling index is category, don't check dtype of others
        try:
            codes = np.concatenate([self._is_dtype_compat(c).codes for c in to_concat])
        except TypeError:
            # not all to_concat elements are among our categories (or NA)
            from pandas.core.dtypes.concat import concat_compat

            res = concat_compat(to_concat)
            return Index(res, name=name)
        else:
            cat = self._data._from_backing_data(codes)
            return type(self)._simple_new(cat, name=name)

    def _delegate_method(self, name: str, *args, **kwargs):
        """method delegation to the ._values"""
        method = getattr(self._values, name)
        if "inplace" in kwargs:
            raise ValueError("cannot use inplace with CategoricalIndex")
        res = method(*args, **kwargs)
        if is_scalar(res):
            return res
        return CategoricalIndex(res, name=self.name)
