import warnings

cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.hashtable cimport HashTable
from pandas._libs.tslibs.nattype cimport c_NaT as NaT
from pandas._libs.tslibs.period cimport is_period_object
from pandas._libs.tslibs.timedeltas cimport _Timedelta
from pandas._libs.tslibs.timestamps cimport _Timestamp

from pandas._libs import (
    algos,
    hashtable as _hash,
)
from pandas._libs.missing import checknull


cdef inline bint is_definitely_invalid_key(object val):
    try:
        hash(val)
    except TypeError:
        return True
    return False


# Don't populate hash tables in monotonic indexes larger than this
_SIZE_CUTOFF = 1_000_000


@cython.freelist(32)
cdef class IndexEngine:

    cdef readonly:
        object vgetter
        HashTable mapping
        bint over_size_threshold

    cdef:
        bint unique, monotonic_inc, monotonic_dec
        bint need_monotonic_check, need_unique_check

    def __init__(self, vgetter, n):
        self.vgetter = vgetter

        self.over_size_threshold = n >= _SIZE_CUTOFF
        self.clear_mapping()

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        self._ensure_mapping_populated()
        return val in self.mapping

    cpdef get_loc(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t loc

        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(val)
            values = self._get_index_values()

            self._check_type(val)
            try:
                loc = _bin_search(values, val)  # .searchsorted(val, side='left')
            except TypeError:
                # GH#35788 e.g. val=None with float64 values
                raise KeyError(val)
            if loc >= len(values):
                raise KeyError(val)
            if values[loc] != val:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(val)

        self._check_type(val)

        try:
            return self.mapping.get_item(val)
        except (TypeError, ValueError, OverflowError):
            # GH#41775 OverflowError e.g. if we are uint64 and val is -1
            raise KeyError(val)

    cdef inline _get_loc_duplicates(self, object val):
        # -> Py_ssize_t | slice | ndarray[bool]
        cdef:
            Py_ssize_t diff

        if self.is_monotonic_increasing:
            values = self._get_index_values()
            try:
                left = values.searchsorted(val, side='left')
                right = values.searchsorted(val, side='right')
            except TypeError:
                # e.g. GH#29189 get_loc(None) with a Float64Index
                raise KeyError(val)

            diff = right - left
            if diff == 0:
                raise KeyError(val)
            elif diff == 1:
                return left
            else:
                return slice(left, right)

        return self._maybe_get_bool_indexer(val)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer

        indexer = self._get_index_values() == val
        return self._unpack_bool_indexer(indexer, val)

    cdef _unpack_bool_indexer(self,
                              ndarray[uint8_t, ndim=1, cast=True] indexer,
                              object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[intp_t, ndim=1] found
            int count

        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)

    def sizeof(self, deep: bool = False) -> int:
        """ return the sizeof our mapping """
        if not self.is_mapping_populated:
            return 0
        return self.mapping.sizeof(deep=deep)

    def __sizeof__(self) -> int:
        return self.sizeof()

    @property
    def is_unique(self) -> bool:
        if self.need_unique_check:
            self._do_unique_check()

        return self.unique == 1

    cdef inline _do_unique_check(self):

        # this de-facto the same
        self._ensure_mapping_populated()

    @property
    def is_monotonic_increasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_inc == 1

    @property
    def is_monotonic_decreasing(self) -> bool:
        if self.need_monotonic_check:
            self._do_monotonic_check()

        return self.monotonic_dec == 1

    cdef inline _do_monotonic_check(self):
        cdef:
            bint is_unique
        try:
            values = self._get_index_values()
            self.monotonic_inc, self.monotonic_dec, is_unique = \
                self._call_monotonic(values)
        except TypeError:
            self.monotonic_inc = 0
            self.monotonic_dec = 0
            is_unique = 0

        self.need_monotonic_check = 0

        # we can only be sure of uniqueness if is_unique=1
        if is_unique:
            self.unique = 1
            self.need_unique_check = 0

    cdef _get_index_values(self):
        return self.vgetter()

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=False)

    def get_backfill_indexer(self, other: np.ndarray, limit=None) -> np.ndarray:
        return algos.backfill(self._get_index_values(), other, limit=limit)

    def get_pad_indexer(self, other: np.ndarray, limit=None) -> np.ndarray:
        return algos.pad(self._get_index_values(), other, limit=limit)

    cdef _make_hash_table(self, Py_ssize_t n):
        raise NotImplementedError

    cdef _check_type(self, object val):
        hash(val)

    @property
    def is_mapping_populated(self) -> bool:
        return self.mapping is not None

    cdef inline _ensure_mapping_populated(self):
        # this populates the mapping
        # if its not already populated
        # also satisfies the need_unique_check

        if not self.is_mapping_populated:

            values = self._get_index_values()
            self.mapping = self._make_hash_table(len(values))
            self._call_map_locations(values)

            if len(self.mapping) == len(values):
                self.unique = 1

        self.need_unique_check = 0

    cdef void _call_map_locations(self, ndarray values):
        self.mapping.map_locations(values)

    def clear_mapping(self):
        self.mapping = None
        self.need_monotonic_check = 1
        self.need_unique_check = 1

        self.unique = 0
        self.monotonic_inc = 0
        self.monotonic_dec = 0

    def get_indexer(self, ndarray values) -> np.ndarray:
        self._ensure_mapping_populated()
        return self.mapping.lookup(values)

    def get_indexer_non_unique(self, ndarray targets):
        """
        Return an indexer suitable for taking from a non unique index
        return the labels in the same order as the target
        and a missing indexer into the targets (which correspond
        to the -1 indices in the results

        Returns
        -------
        indexer : np.ndarray[np.intp]
        missing : np.ndarray[np.intp]
        """
        cdef:
            ndarray values, x
            ndarray[intp_t] result, missing
            set stargets, remaining_stargets
            dict d = {}
            object val
            int count = 0, count_missing = 0
            Py_ssize_t i, j, n, n_t, n_alloc

        self._ensure_mapping_populated()
        values = np.array(self._get_index_values(), copy=False)
        stargets = set(targets)
        n = len(values)
        n_t = len(targets)
        if n > 10_000:
            n_alloc = 10_000
        else:
            n_alloc = n

        result = np.empty(n_alloc, dtype=np.intp)
        missing = np.empty(n_t, dtype=np.intp)

        # map each starget to its position in the index
        if stargets and len(stargets) < 5 and self.is_monotonic_increasing:
            # if there are few enough stargets and the index is monotonically
            # increasing, then use binary search for each starget
            remaining_stargets = set()
            for starget in stargets:
                try:
                    start = values.searchsorted(starget, side='left')
                    end = values.searchsorted(starget, side='right')
                except TypeError:  # e.g. if we tried to search for string in int array
                    remaining_stargets.add(starget)
                else:
                    if start != end:
                        d[starget] = list(range(start, end))

            stargets = remaining_stargets

        if stargets:
            # otherwise, map by iterating through all items in the index
            for i in range(n):
                val = values[i]
                if val in stargets:
                    if val not in d:
                        d[val] = []
                    d[val].append(i)

        for i in range(n_t):
            val = targets[i]

            # found
            if val in d:
                for j in d[val]:

                    # realloc if needed
                    if count >= n_alloc:
                        n_alloc += 10_000
                        result = np.resize(result, n_alloc)

                    result[count] = j
                    count += 1

            # value not found
            else:

                if count >= n_alloc:
                    n_alloc += 10_000
                    result = np.resize(result, n_alloc)
                result[count] = -1
                count += 1
                missing[count_missing] = i
                count_missing += 1

        return result[0:count], missing[0:count_missing]


cdef Py_ssize_t _bin_search(ndarray values, object val) except -1:
    cdef:
        Py_ssize_t mid = 0, lo = 0, hi = len(values) - 1
        object pval

    if hi == 0 or (hi > 0 and val > values[hi]):
        return len(values)

    while lo < hi:
        mid = (lo + hi) // 2
        pval = values[mid]
        if val < pval:
            hi = mid
        elif val > pval:
            lo = mid + 1
        else:
            while mid > 0 and val == values[mid - 1]:
                mid -= 1
            return mid

    if val <= values[mid]:
        return mid
    else:
        return mid + 1


cdef class ObjectEngine(IndexEngine):
    """
    Index Engine for use with object-dtype Index, namely the base class Index.
    """
    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.PyObjectHashTable(n)


cdef class DatetimeEngine(Int64Engine):

    cdef str _get_box_dtype(self):
        return 'M8[ns]'

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        # NB: caller is responsible for ensuring tzawareness compat
        #  before we get here
        if not (isinstance(scalar, _Timestamp) or scalar is NaT):
            raise TypeError(scalar)
        return scalar.value

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        cdef:
            int64_t loc, conv

        conv = self._unbox_scalar(val)
        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(conv)
            values = self._get_index_values()
            loc = values.searchsorted(conv, side='left')
            return values[loc] == conv

        self._ensure_mapping_populated()
        return conv in self.mapping

    cdef _get_index_values(self):
        return self.vgetter().view('i8')

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)

    cpdef get_loc(self, object val):
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Timestamp or NaT (Timedelta or NaT for TimedeltaEngine)

        cdef:
            int64_t loc
        if is_definitely_invalid_key(val):
            raise TypeError(f"'{val}' is an invalid key")

        try:
            conv = self._unbox_scalar(val)
        except TypeError:
            raise KeyError(val)

        # Welcome to the spaghetti factory
        if self.over_size_threshold and self.is_monotonic_increasing:
            if not self.is_unique:
                return self._get_loc_duplicates(conv)
            values = self._get_index_values()

            loc = values.searchsorted(conv, side='left')

            if loc == len(values) or values[loc] != conv:
                raise KeyError(val)
            return loc

        self._ensure_mapping_populated()
        if not self.unique:
            return self._get_loc_duplicates(conv)

        try:
            return self.mapping.get_item(conv)
        except KeyError:
            raise KeyError(val)

    def get_indexer_non_unique(self, ndarray targets):
        # we may get datetime64[ns] or timedelta64[ns], cast these to int64
        return super().get_indexer_non_unique(targets.view("i8"))

    def get_indexer(self, ndarray values) -> np.ndarray:
        self._ensure_mapping_populated()
        if values.dtype != self._get_box_dtype():
            return np.repeat(-1, len(values)).astype(np.intp)
        values = np.asarray(values).view('i8')
        return self.mapping.lookup(values)

    def get_pad_indexer(self, other: np.ndarray, limit=None) -> np.ndarray:
        if other.dtype != self._get_box_dtype():
            return np.repeat(-1, len(other)).astype(np.intp)
        other = np.asarray(other).view('i8')
        return algos.pad(self._get_index_values(), other, limit=limit)

    def get_backfill_indexer(self, other: np.ndarray, limit=None) -> np.ndarray:
        if other.dtype != self._get_box_dtype():
            return np.repeat(-1, len(other)).astype(np.intp)
        other = np.asarray(other).view('i8')
        return algos.backfill(self._get_index_values(), other, limit=limit)


cdef class TimedeltaEngine(DatetimeEngine):

    cdef str _get_box_dtype(self):
        return 'm8[ns]'

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        if not (isinstance(scalar, _Timedelta) or scalar is NaT):
            raise TypeError(scalar)
        return scalar.value


cdef class PeriodEngine(Int64Engine):

    cdef int64_t _unbox_scalar(self, scalar) except? -1:
        if scalar is NaT:
            return scalar.value
        if is_period_object(scalar):
            # NB: we assume that we have the correct freq here.
            return scalar.ordinal
        raise TypeError(scalar)

    cpdef get_loc(self, object val):
        # NB: the caller is responsible for ensuring that we are called
        #  with either a Period or NaT
        cdef:
            int64_t conv

        try:
            conv = self._unbox_scalar(val)
        except TypeError:
            raise KeyError(val)

        return Int64Engine.get_loc(self, conv)

    cdef _get_index_values(self):
        return super(PeriodEngine, self).vgetter().view("i8")

    cdef _call_monotonic(self, values):
        return algos.is_monotonic(values, timelike=True)


cdef class BaseMultiIndexCodesEngine:
    """
    Base class for MultiIndexUIntEngine and MultiIndexPyIntEngine, which
    represent each label in a MultiIndex as an integer, by juxtaposing the bits
    encoding each level, with appropriate offsets.

    For instance: if 3 levels have respectively 3, 6 and 1 possible values,
    then their labels can be represented using respectively 2, 3 and 1 bits,
    as follows:
     _ _ _ _____ _ __ __ __
    |0|0|0| ... |0| 0|a1|a0| -> offset 0 (first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0|b2|b1|b0| -> offset 2 (bits required for first level)
     — — — ————— — —— —— ——
    |0|0|0| ... |0| 0| 0|c0| -> offset 5 (bits required for first two levels)
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾
    and the resulting unsigned integer representation will be:
     _ _ _ _____ _ __ __ __ __ __ __
    |0|0|0| ... |0|c0|b2|b1|b0|a1|a0|
     ‾ ‾ ‾ ‾‾‾‾‾ ‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾ ‾‾

    Offsets are calculated at initialization, labels are transformed by method
    _codes_to_ints.

    Keys are located by first locating each component against the respective
    level, then locating (the integer representation of) codes.
    """
    def __init__(self, object levels, object labels,
                 ndarray[uint64_t, ndim=1] offsets):
        """
        Parameters
        ----------
        levels : list-like of numpy arrays
            Levels of the MultiIndex.
        labels : list-like of numpy arrays of integer dtype
            Labels of the MultiIndex.
        offsets : numpy array of uint64 dtype
            Pre-calculated offsets, one for each level of the index.
        """
        self.levels = levels
        self.offsets = offsets

        # Transform labels in a single array, and add 1 so that we are working
        # with positive integers (-1 for NaN becomes 0):
        codes = (np.array(labels, dtype='int64').T + 1).astype('uint64',
                                                               copy=False)

        # Map each codes combination in the index to an integer unambiguously
        # (no collisions possible), based on the "offsets", which describe the
        # number of bits to switch labels for each level:
        lab_ints = self._codes_to_ints(codes)

        # Initialize underlying index (e.g. libindex.UInt64Engine) with
        # integers representing labels: we will use its get_loc and get_indexer
        self._base.__init__(self, lambda: lab_ints, len(lab_ints))

    def _codes_to_ints(self, ndarray[uint64_t] codes) -> np.ndarray:
        raise NotImplementedError("Implemented by subclass")

    def _extract_level_codes(self, ndarray[object] target) -> np.ndarray:
        """
        Map the requested list of (tuple) keys to their integer representations
        for searching in the underlying integer index.

        Parameters
        ----------
        target : ndarray[object]
            Each key is a tuple, with a label for each level of the index.

        Returns
        ------
        int_keys : 1-dimensional array of dtype uint64 or object
            Integers representing one combination each
        """
        level_codes = [lev.get_indexer(codes) + 1 for lev, codes
                       in zip(self.levels, zip(*target))]
        return self._codes_to_ints(np.array(level_codes, dtype='uint64').T)

    def get_indexer(self, ndarray[object] target) -> np.ndarray:
        """
        Returns an array giving the positions of each value of `target` in
        `self.values`, where -1 represents a value in `target` which does not
        appear in `self.values`

        Parameters
        ----------
        target : ndarray[object]
            Each key is a tuple, with a label for each level of the index

        Returns
        -------
        np.ndarray[intp_t, ndim=1] of the indexer of `target` into
        `self.values`
        """
        lab_ints = self._extract_level_codes(target)
        return self._base.get_indexer(self, lab_ints)

    def get_indexer_with_fill(self, ndarray target, ndarray values,
                              str method, object limit) -> np.ndarray:
        """
        Returns an array giving the positions of each value of `target` in
        `values`, where -1 represents a value in `target` which does not
        appear in `values`

        If `method` is "backfill" then the position for a value in `target`
        which does not appear in `values` is that of the next greater value
        in `values` (if one exists), and -1 if there is no such value.

        Similarly, if the method is "pad" then the position for a value in
        `target` which does not appear in `values` is that of the next smaller
        value in `values` (if one exists), and -1 if there is no such value.

        Parameters
        ----------
        target: ndarray[object] of tuples
            need not be sorted, but all must have the same length, which must be
            the same as the length of all tuples in `values`
        values : ndarray[object] of tuples
            must be sorted and all have the same length.  Should be the set of
            the MultiIndex's values.
        method: string
            "backfill" or "pad"
        limit: int or None
            if provided, limit the number of fills to this value

        Returns
        -------
        np.ndarray[intp_t, ndim=1] of the indexer of `target` into `values`,
        filled with the `method` (and optionally `limit`) specified
        """
        assert method in ("backfill", "pad")
        cdef:
            int64_t i, j, next_code
            int64_t num_values, num_target_values
            ndarray[int64_t, ndim=1] target_order
            ndarray[object, ndim=1] target_values
            ndarray[int64_t, ndim=1] new_codes, new_target_codes
            ndarray[intp_t, ndim=1] sorted_indexer

        target_order = np.argsort(target).astype('int64')
        target_values = target[target_order]
        num_values, num_target_values = len(values), len(target_values)
        new_codes, new_target_codes = (
            np.empty((num_values,)).astype('int64'),
            np.empty((num_target_values,)).astype('int64'),
        )

        # `values` and `target_values` are both sorted, so we walk through them
        # and memoize the (ordered) set of indices in the (implicit) merged-and
        # sorted list of the two which belong to each of them
        # the effect of this is to create a factorization for the (sorted)
        # merger of the index values, where `new_codes` and `new_target_codes`
        # are the subset of the factors which appear in `values` and `target`,
        # respectively
        i, j, next_code = 0, 0, 0
        while i < num_values and j < num_target_values:
            val, target_val = values[i], target_values[j]
            if val <= target_val:
                new_codes[i] = next_code
                i += 1
            if target_val <= val:
                new_target_codes[j] = next_code
                j += 1
            next_code += 1

        # at this point, at least one should have reached the end
        # the remaining values of the other should be added to the end
        assert i == num_values or j == num_target_values
        while i < num_values:
            new_codes[i] = next_code
            i += 1
            next_code += 1
        while j < num_target_values:
            new_target_codes[j] = next_code
            j += 1
            next_code += 1

        # get the indexer, and undo the sorting of `target.values`
        algo = algos.backfill if method == "backfill" else algos.pad
        sorted_indexer = algo(new_codes, new_target_codes, limit=limit)
        return sorted_indexer[np.argsort(target_order)]

    def get_loc(self, object key):
        if is_definitely_invalid_key(key):
            raise TypeError(f"'{key}' is an invalid key")
        if not isinstance(key, tuple):
            raise KeyError(key)
        try:
            indices = [0 if checknull(v) else lev.get_loc(v) + 1
                       for lev, v in zip(self.levels, key)]
        except KeyError:
            raise KeyError(key)

        # Transform indices into single integer:
        lab_int = self._codes_to_ints(np.array(indices, dtype='uint64'))

        return self._base.get_loc(self, lab_int)

    def get_indexer_non_unique(self, ndarray[object] target):

        lab_ints = self._extract_level_codes(target)
        indexer = self._base.get_indexer_non_unique(self, lab_ints)

        return indexer

    def __contains__(self, val: object) -> bool:
        # We assume before we get here:
        #  - val is hashable
        # Default __contains__ looks in the underlying mapping, which in this
        # case only contains integer representations.
        try:
            self.get_loc(val)
            return True
        except (KeyError, TypeError, ValueError):
            return False


# Generated from template.
include "index_class_helper.pxi"
