""" miscellaneous sorting / groupby utilities """
from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Iterable,
    Sequence,
)

import numpy as np

from pandas._libs import (
    algos,
    hashtable,
    lib,
)
from pandas._libs.hashtable import unique_label_indices
from pandas._typing import (
    IndexKeyFunc,
    Shape,
)

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_extension_array_dtype,
)
from pandas.core.dtypes.generic import (
    ABCMultiIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.missing import isna

from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from pandas import MultiIndex
    from pandas.core.indexes.base import Index


def get_indexer_indexer(
    target: Index,
    level: str | int | list[str] | list[int],
    ascending: Sequence[bool | int] | bool | int,
    kind: str,
    na_position: str,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> np.ndarray | None:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
    level : int or level name or list of ints or list of level names
    ascending : bool or list of bools, default True
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    na_position : {'first', 'last'}, default 'last'
    sort_remaining : bool, default True
    key : callable, optional

    Returns
    -------
    Optional[ndarray]
        The indexer for the new index.
    """

    target = ensure_key_mapped(target, key, levels=level)
    target = target._sort_levels_monotonic()

    if level is not None:
        _, indexer = target.sortlevel(
            level, ascending=ascending, sort_remaining=sort_remaining
        )
    elif isinstance(target, ABCMultiIndex):
        indexer = lexsort_indexer(
            target._get_codes_for_sorting(), orders=ascending, na_position=na_position
        )
    else:
        # Check monotonic-ness before sort an index (GH 11080)
        if (ascending and target.is_monotonic_increasing) or (
            not ascending and target.is_monotonic_decreasing
        ):
            return None

        indexer = nargsort(
            target, kind=kind, ascending=ascending, na_position=na_position
        )
    return indexer


def get_group_index(labels, shape: Shape, sort: bool, xnull: bool):
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If the ranks of returned ids should match lexical ranks of labels
    xnull : bool
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through.

    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.

    Notes
    -----
    The length of `labels` and `shape` must be identical.
    """

    def _int64_cut_off(shape) -> int:
        acc = 1
        for i, mul in enumerate(shape):
            acc *= int(mul)
            if not acc < lib.i8max:
                return i
        return len(shape)

    def maybe_lift(lab, size) -> tuple[np.ndarray, int]:
        # promote nan values (assigned -1 label in lab array)
        # so that all output values are non-negative
        return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)

    labels = [ensure_int64(x) for x in labels]
    lshape = list(shape)
    if not xnull:
        for i, (lab, size) in enumerate(zip(labels, shape)):
            lab, size = maybe_lift(lab, size)
            labels[i] = lab
            lshape[i] = size

    labels = list(labels)

    # Iteratively process all the labels in chunks sized so less
    # than lib.i8max unique int ids will be required for each chunk
    while True:
        # how many levels can be done without overflow:
        nlev = _int64_cut_off(lshape)

        # compute flat ids for the first `nlev` levels
        stride = np.prod(lshape[1:nlev], dtype="i8")
        out = stride * labels[0].astype("i8", subok=False, copy=False)

        for i in range(1, nlev):
            if lshape[i] == 0:
                stride = np.int64(0)
            else:
                stride //= lshape[i]
            out += labels[i] * stride

        if xnull:  # exclude nulls
            mask = labels[0] == -1
            for lab in labels[1:nlev]:
                mask |= lab == -1
            out[mask] = -1

        if nlev == len(lshape):  # all levels done!
            break

        # compress what has been done so far in order to avoid overflow
        # to retain lexical ranks, obs_ids should be sorted
        comp_ids, obs_ids = compress_group_index(out, sort=sort)

        labels = [comp_ids] + labels[nlev:]
        lshape = [len(obs_ids)] + lshape[nlev:]

    return out


def get_compressed_ids(labels, sizes: Shape) -> tuple[np.ndarray, np.ndarray]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)


def is_int64_overflow_possible(shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


def decons_group_index(comp_labels, shape):
    # reconstruct labels
    if is_int64_overflow_possible(shape):
        # at some point group indices are factorized,
        # and may not be deconstructed here! wrong path!
        raise ValueError("cannot deconstruct factorized group indices!")

    label_list = []
    factor = 1
    y = 0
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (x - y) % (factor * shape[i]) // factor
        np.putmask(labels, comp_labels < 0, -1)
        label_list.append(labels)
        y = labels * factor
        factor *= shape[i]
    return label_list[::-1]


def decons_obs_group_ids(comp_ids: np.ndarray, obs_ids, shape, labels, xnull: bool):
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
    """
    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype="i8")
        shape = np.asarray(shape, dtype="i8") + lift

    if not is_int64_overflow_possible(shape):
        # obs ids are deconstructable! take the fast route!
        out = decons_group_index(obs_ids, shape)
        return out if xnull or not lift.any() else [x - y for x, y in zip(out, lift)]

    # TODO: unique_label_indices only used here, should take ndarray[np.intp]
    indexer = unique_label_indices(ensure_int64(comp_ids))
    return [lab[indexer].astype(np.intp, subok=False, copy=True) for lab in labels]


def indexer_from_factorized(labels, shape: Shape, compress: bool = True) -> np.ndarray:
    # returned ndarray is np.intp
    ids = get_group_index(labels, shape, sort=True, xnull=False)

    if not compress:
        ngroups = (ids.size and ids.max()) + 1
    else:
        ids, obs = compress_group_index(ids, sort=True)
        ngroups = len(obs)

    return get_group_index_sorter(ids, ngroups)


def lexsort_indexer(
    keys, orders=None, na_position: str = "last", key: Callable | None = None
) -> np.ndarray:
    """
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : sequence of arrays
        Sequence of ndarrays to be sorted by the indexer
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. if bool, applied to all
        elements as above. if None, defaults to True.
    na_position : {'first', 'last'}, default 'last'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting

        .. versionadded:: 1.0.0

    Returns
    -------
    np.ndarray[np.intp]
    """
    from pandas.core.arrays import Categorical

    labels = []
    shape = []
    if isinstance(orders, bool):
        orders = [orders] * len(keys)
    elif orders is None:
        orders = [True] * len(keys)

    keys = [ensure_key_mapped(k, key) for k in keys]

    for k, order in zip(keys, orders):
        cat = Categorical(k, ordered=True)

        if na_position not in ["last", "first"]:
            raise ValueError(f"invalid na_position: {na_position}")

        n = len(cat.categories)
        codes = cat.codes.copy()

        mask = cat.codes == -1
        if order:  # ascending
            if na_position == "last":
                codes = np.where(mask, n, codes)
            elif na_position == "first":
                codes += 1
        else:  # not order means descending
            if na_position == "last":
                codes = np.where(mask, n, n - codes - 1)
            elif na_position == "first":
                codes = np.where(mask, 0, n - codes)
        if mask.any():
            n += 1

        shape.append(n)
        labels.append(codes)

    return indexer_from_factorized(labels, tuple(shape))


def nargsort(
    items,
    kind: str = "quicksort",
    ascending: bool = True,
    na_position: str = "last",
    key: Callable | None = None,
    mask: np.ndarray | None = None,
):
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    kind : str, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """

    if key is not None:
        items = ensure_key_mapped(items, key)
        return nargsort(
            items,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            key=None,
            mask=mask,
        )

    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending=ascending)  # TODO: test coverage with key?
    elif not isinstance(items, ABCMultiIndex):
        items = extract_array(items)
    if mask is None:
        mask = np.asarray(isna(items))  # TODO: does this exclude MultiIndex too?

    if is_extension_array_dtype(items):
        return items.argsort(ascending=ascending, kind=kind, na_position=na_position)
    else:
        items = np.asanyarray(items)

    idx = np.arange(len(items))
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]

    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    # Finally, place the NaNs at the end or the beginning according to
    # na_position
    if na_position == "last":
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == "first":
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f"invalid na_position: {na_position}")
    return ensure_platform_int(indexer)


def nargminmax(values, method: str, axis: int = 0):
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
    method : {"argmax", "argmin"}
    axis : int, default 0

    Returns
    -------
    int
    """
    assert method in {"argmax", "argmin"}
    func = np.argmax if method == "argmax" else np.argmin

    mask = np.asarray(isna(values))
    values = values._values_for_argsort()

    if values.ndim > 1:
        if mask.any():
            if axis == 1:
                zipped = zip(values, mask)
            else:
                zipped = zip(values.T, mask.T)
            return np.array([_nanargminmax(v, m, func) for v, m in zipped])
        return func(values, axis=axis)

    return _nanargminmax(values, mask, func)


def _nanargminmax(values, mask, func) -> int:
    """
    See nanargminmax.__doc__.
    """
    idx = np.arange(values.shape[0])
    non_nans = values[~mask]
    non_nan_idx = idx[~mask]

    return non_nan_idx[func(non_nans)]


def _ensure_key_mapped_multiindex(
    index: MultiIndex, key: Callable, level=None
) -> MultiIndex:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None). Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        Index to which to apply the key function on the
        specified levels.
    key : Callable
        Function that takes an Index and returns an Index of
        the same shape. This key is applied to each level
        separately. The name of the level can be used to
        distinguish different levels for application.
    level : list-like, int or str, default None
        Level or list of levels to apply the key function to.
        If None, key function is applied to all levels. Other
        levels are left unchanged.

    Returns
    -------
    labels : MultiIndex
        Resulting MultiIndex with modified levels.
    """

    if level is not None:
        if isinstance(level, (str, int)):
            sort_levels = [level]
        else:
            sort_levels = level

        sort_levels = [index._get_level_number(lev) for lev in sort_levels]
    else:
        sort_levels = list(range(index.nlevels))  # satisfies mypy

    mapped = [
        ensure_key_mapped(index._get_level_values(level), key)
        if level in sort_levels
        else index._get_level_values(level)
        for level in range(index.nlevels)
    ]

    return type(index).from_arrays(mapped)


def ensure_key_mapped(values, key: Callable | None, levels=None):
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    from pandas.core.indexes.api import Index

    if not key:
        return values

    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)

    result = key(values.copy())
    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change the shape of the array."
        )

    try:
        if isinstance(
            values, Index
        ):  # convert to a new Index subclass, not necessarily the same
            result = Index(result)
        else:
            type_of_values = type(values)
            result = type_of_values(result)  # try to revert to original type otherwise
    except TypeError:
        raise TypeError(
            f"User-provided `key` function returned an invalid type {type(result)} \
            which could not be converted to {type(values)}."
        )

    return result


def get_flattened_list(
    comp_ids: np.ndarray,  # np.ndarray[np.intp]
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> list[tuple]:
    """Map compressed group id -> key tuple."""
    comp_ids = comp_ids.astype(np.int64, copy=False)
    arrays: DefaultDict[int, list[int]] = defaultdict(list)
    for labs, level in zip(labels, levels):
        table = hashtable.Int64HashTable(ngroups)
        table.map(comp_ids, labs.astype(np.int64, copy=False))
        for i in range(ngroups):
            arrays[i].append(level[table.get_item(i)])
    return [tuple(array) for array in arrays.values()]


def get_indexer_dict(
    label_list: list[np.ndarray], keys: list[Index]
) -> dict[str | tuple, np.ndarray]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
    shape = [len(x) for x in keys]

    group_index = get_group_index(label_list, tuple(shape), sort=True, xnull=True)
    if np.all(group_index == -1):
        # Short-circuit, lib.indices_fast will return the same
        return {}
    ngroups = (
        ((group_index.size and group_index.max()) + 1)
        if is_int64_overflow_possible(shape)
        else np.prod(shape, dtype="i8")
    )

    sorter = get_group_index_sorter(group_index, ngroups)

    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)

    return lib.indices_fast(sorter, group_index, keys, sorted_labels)


# ----------------------------------------------------------------------
# sorting levels...cleverly?


def get_group_index_sorter(
    group_index: np.ndarray, ngroups: int | None = None
) -> np.ndarray:
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        signed integer dtype
    ngroups : int or None, default None

    Returns
    -------
    np.ndarray[np.intp]
    """
    if ngroups is None:
        ngroups = 1 + group_index.max()
    count = len(group_index)
    alpha = 0.0  # taking complexities literally; there may be
    beta = 1.0  # some room for fine-tuning these parameters
    do_groupsort = count > 0 and ((alpha + beta * ngroups) < (count * np.log(count)))
    if do_groupsort:
        sorter, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index),
            ngroups,
        )
        # sorter _should_ already be intp, but mypy is not yet able to verify
    else:
        sorter = group_index.argsort(kind="mergesort")
    return ensure_platform_int(sorter)


def compress_group_index(
    group_index: np.ndarray, sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    size_hint = len(group_index)
    table = hashtable.Int64HashTable(size_hint)

    group_index = ensure_int64(group_index)

    # note, group labels come out ascending (ie, 1,2,3 etc)
    comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

    if sort and len(obs_group_ids) > 0:
        obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)


def _reorder_by_uniques(
    uniques: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    uniques : np.ndarray[np.int64]
    labels : np.ndarray[np.intp]

    Returns
    -------
    np.ndarray[np.int64]
    np.ndarray[np.intp]
    """
    # sorter is index where elements ought to go
    sorter = uniques.argsort()

    # reverse_indexer is where elements came from
    reverse_indexer = np.empty(len(sorter), dtype=np.intp)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    mask = labels < 0

    # move labels to right locations (ie, unsort ascending labels)
    labels = reverse_indexer.take(labels)
    np.putmask(labels, mask, -1)

    # sort observed ids
    uniques = uniques.take(sorter)

    return uniques, labels
