
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

from pandas._libs.util cimport (
    is_array,
    set_array_not_contiguous,
)

from pandas._libs.lib import is_scalar


cdef cnp.dtype _dtype_obj = np.dtype("object")


cpdef check_result_array(object obj, object dtype):
    # Our operation is supposed to be an aggregation/reduction. If
    #  it returns an ndarray, this likely means an invalid operation has
    #  been passed. See test_apply_without_aggregation, test_agg_must_agg
    if is_array(obj):
        if dtype != _dtype_obj:
            # If it is object dtype, the function can be a reduction/aggregation
            #  and still return an ndarray e.g. test_agg_over_numpy_arrays
            raise ValueError("Must produce aggregated value")


cdef class _BaseGrouper:
    cdef _check_dummy(self, object dummy):
        # both values and index must be an ndarray!

        values = dummy.values
        # GH 23683: datetimetz types are equivalent to datetime types here
        if (dummy.dtype != self.arr.dtype
                and values.dtype != self.arr.dtype):
            raise ValueError('Dummy array must be same dtype')
        if is_array(values) and not values.flags.contiguous:
            # e.g. Categorical has no `flags` attribute
            values = values.copy()
        index = dummy.index.values
        if not index.flags.contiguous:
            index = index.copy()

        return values, index

    cdef _init_dummy_series_and_index(self, Slider islider, Slider vslider):
        """
        Create Series and Index objects that we will alter in-place while iterating.
        """
        cached_index = self.ityp(islider.buf, dtype=self.idtype)
        cached_series = self.typ(
            vslider.buf, dtype=vslider.buf.dtype, index=cached_index, name=self.name
        )
        return cached_index, cached_series

    cdef inline _update_cached_objs(self, object cached_series, object cached_index,
                                    Slider islider, Slider vslider):
        # See the comment in indexes/base.py about _index_data.
        # We need this for EA-backed indexes that have a reference
        # to a 1-d ndarray like datetime / timedelta / period.
        cached_index._engine.clear_mapping()
        cached_index._cache.clear()  # e.g. inferred_freq must go
        cached_series._mgr.set_values(vslider.buf)

    cdef inline object _apply_to_group(self,
                                       object cached_series, object cached_index,
                                       bint initialized):
        """
        Call self.f on our new group, then update to the next group.
        """
        cdef:
            object res

        # NB: we assume that _update_cached_objs has already cleared cleared
        #  the cache and engine mapping
        res = self.f(cached_series)
        res = extract_result(res)
        if not initialized:
            # On the first pass, we check the output shape to see
            #  if this looks like a reduction.
            initialized = True
            check_result_array(res, cached_series.dtype)

        return res, initialized


cdef class SeriesBinGrouper(_BaseGrouper):
    """
    Performs grouping operation according to bin edges, rather than labels
    """
    cdef:
        Py_ssize_t nresults, ngroups

    cdef public:
        ndarray bins  # ndarray[int64_t]
        ndarray arr, index, dummy_arr, dummy_index
        object values, f, typ, ityp, name, idtype

    def __init__(self, object series, object f, ndarray[int64_t] bins):

        assert len(bins) > 0  # otherwise we get IndexError in get_result

        self.bins = bins
        self.f = f

        values = series.values
        if is_array(values) and not values.flags.c_contiguous:
            # e.g. Categorical has no `flags` attribute
            values = values.copy('C')
        self.arr = values
        self.typ = series._constructor
        self.ityp = series.index._constructor
        self.idtype = series.index.dtype
        self.index = series.index.values
        self.name = series.name

        dummy = series.iloc[:0]
        self.dummy_arr, self.dummy_index = self._check_dummy(dummy)

        # kludge for #1688
        if len(bins) > 0 and bins[-1] == len(series):
            self.ngroups = len(bins)
        else:
            # TODO: not reached except in test_series_bin_grouper directly
            #  constructing SeriesBinGrouper; can we rule this case out?
            self.ngroups = len(bins) + 1

    def get_result(self):
        cdef:
            ndarray arr, result
            ndarray[int64_t] counts
            Py_ssize_t i, n, group_size, start, end
            object res
            bint initialized = 0
            Slider vslider, islider
            object cached_series = None, cached_index = None

        counts = np.zeros(self.ngroups, dtype=np.int64)

        if self.ngroups > 0:
            counts[0] = self.bins[0]
            for i in range(1, self.ngroups):
                if i == self.ngroups - 1:
                    counts[i] = len(self.arr) - self.bins[i - 1]
                else:
                    counts[i] = self.bins[i] - self.bins[i - 1]

        group_size = 0
        n = len(self.arr)

        vslider = Slider(self.arr, self.dummy_arr)
        islider = Slider(self.index, self.dummy_index)

        result = np.empty(self.ngroups, dtype='O')

        cached_index, cached_series = self._init_dummy_series_and_index(
            islider, vslider
        )

        start = 0
        try:
            for i in range(self.ngroups):
                group_size = counts[i]
                end = start + group_size

                islider.move(start, end)
                vslider.move(start, end)

                self._update_cached_objs(
                    cached_series, cached_index, islider, vslider)

                res, initialized = self._apply_to_group(cached_series, cached_index,
                                                        initialized)
                start += group_size

                result[i] = res

        finally:
            # so we don't free the wrong memory
            islider.reset()
            vslider.reset()

        return result, counts


cdef class SeriesGrouper(_BaseGrouper):
    """
    Performs generic grouping operation while avoiding ndarray construction
    overhead
    """
    cdef:
        Py_ssize_t nresults, ngroups

    cdef public:
        ndarray arr, index, dummy_arr, dummy_index
        object f, labels, values, typ, ityp, name, idtype

    def __init__(self, object series, object f, ndarray[intp_t] labels,
                 Py_ssize_t ngroups):

        if len(series) == 0:
            # get_result would never assign `result`
            raise ValueError("SeriesGrouper requires non-empty `series`")

        self.labels = labels
        self.f = f

        values = series.values
        if is_array(values) and not values.flags.c_contiguous:
            # e.g. Categorical has no `flags` attribute
            values = values.copy('C')
        self.arr = values
        self.typ = series._constructor
        self.ityp = series.index._constructor
        self.idtype = series.index.dtype
        self.index = series.index.values
        self.name = series.name

        dummy = series.iloc[:0]
        self.dummy_arr, self.dummy_index = self._check_dummy(dummy)
        self.ngroups = ngroups

    def get_result(self):
        cdef:
            # Define result to avoid UnboundLocalError
            ndarray arr, result = None
            ndarray[intp_t] labels
            ndarray[int64_t] counts
            Py_ssize_t i, n, group_size, lab, start, end
            object res
            bint initialized = 0
            Slider vslider, islider
            object cached_series = None, cached_index = None

        labels = self.labels
        counts = np.zeros(self.ngroups, dtype=np.int64)
        group_size = 0
        n = len(self.arr)

        vslider = Slider(self.arr, self.dummy_arr)
        islider = Slider(self.index, self.dummy_index)

        result = np.empty(self.ngroups, dtype='O')

        cached_index, cached_series = self._init_dummy_series_and_index(
            islider, vslider
        )

        start = 0
        try:
            for i in range(n):
                group_size += 1

                lab = labels[i]

                if i == n - 1 or lab != labels[i + 1]:
                    if lab == -1:
                        start += group_size
                        group_size = 0
                        continue

                    end = start + group_size
                    islider.move(start, end)
                    vslider.move(start, end)

                    self._update_cached_objs(
                        cached_series, cached_index, islider, vslider)

                    res, initialized = self._apply_to_group(cached_series, cached_index,
                                                            initialized)

                    start += group_size

                    result[lab] = res
                    counts[lab] = group_size
                    group_size = 0

        finally:
            # so we don't free the wrong memory
            islider.reset()
            vslider.reset()

        # We check for empty series in the constructor, so should always
        #  have result initialized by this point.
        assert initialized, "`result` has not been initialized."

        return result, counts


cpdef inline extract_result(object res):
    """ extract the result object, it might be a 0-dim ndarray
        or a len-1 0-dim, or a scalar """
    if hasattr(res, "_values"):
        # Preserve EA
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            # see test_agg_lambda_with_timezone, test_resampler_grouper.py::test_apply
            res = res[0]
    if is_array(res):
        if res.ndim == 1 and len(res) == 1:
            # see test_resampler_grouper.py::test_apply
            res = res[0]
    return res


cdef class Slider:
    """
    Only handles contiguous data for now
    """
    cdef:
        ndarray values, buf
        Py_ssize_t stride
        char *orig_data

    def __init__(self, ndarray values, ndarray buf):
        assert values.ndim == 1
        assert values.dtype == buf.dtype

        if not values.flags.contiguous:
            values = values.copy()

        self.values = values
        self.buf = buf

        self.stride = values.strides[0]
        self.orig_data = self.buf.data

        self.buf.data = self.values.data
        self.buf.strides[0] = self.stride

    cdef move(self, int start, int end):
        """
        For slicing
        """
        self.buf.data = self.values.data + self.stride * start
        self.buf.shape[0] = end - start

    cdef reset(self):
        self.buf.data = self.orig_data
        self.buf.shape[0] = 0


def apply_frame_axis0(object frame, object f, object names,
                      const int64_t[:] starts, const int64_t[:] ends):
    cdef:
        BlockSlider slider
        Py_ssize_t i, n = len(starts)
        list results
        object piece
        dict item_cache

    # We have already checked that we don't have a MultiIndex before calling
    assert frame.index.nlevels == 1

    results = []

    slider = BlockSlider(frame)

    mutated = False
    item_cache = slider.dummy._item_cache
    try:
        for i in range(n):
            slider.move(starts[i], ends[i])

            item_cache.clear()  # ugh
            chunk = slider.dummy
            object.__setattr__(chunk, 'name', names[i])

            piece = f(chunk)

            # Need to infer if low level index slider will cause segfaults
            require_slow_apply = i == 0 and piece is chunk
            try:
                if piece.index is not chunk.index:
                    mutated = True
            except AttributeError:
                # `piece` might not have an index, could be e.g. an int
                pass

            if not is_scalar(piece):
                # Need to copy data to avoid appending references
                try:
                    piece = piece.copy(deep="all")
                except (TypeError, AttributeError):
                    pass

            results.append(piece)

            # If the data was modified inplace we need to
            # take the slow path to not risk segfaults
            # we have already computed the first piece
            if require_slow_apply:
                break
    finally:
        slider.reset()

    return results, mutated


cdef class BlockSlider:
    """
    Only capable of sliding on axis=0
    """
    cdef:
        object frame, dummy, index, block
        list blocks, blk_values
        ndarray orig_blklocs, orig_blknos
        ndarray values
        Slider idx_slider
        char **base_ptrs
        int nblocks
        Py_ssize_t i

    def __init__(self, object frame):
        self.frame = frame
        self.dummy = frame[:0]
        self.index = self.dummy.index

        # GH#35417 attributes we need to restore at each step in case
        #  the function modified them.
        mgr = self.dummy._mgr
        self.orig_blklocs = mgr.blklocs
        self.orig_blknos = mgr.blknos
        self.blocks = [x for x in self.dummy._mgr.blocks]

        self.blk_values = [block.values for block in self.dummy._mgr.blocks]

        for values in self.blk_values:
            set_array_not_contiguous(values)

        self.nblocks = len(self.blk_values)
        # See the comment in indexes/base.py about _index_data.
        # We need this for EA-backed indexes that have a reference to a 1-d
        # ndarray like datetime / timedelta / period.
        self.idx_slider = Slider(
            self.frame.index._index_data, self.dummy.index._index_data)

        self.base_ptrs = <char**>malloc(sizeof(char*) * self.nblocks)
        for i, block in enumerate(self.blk_values):
            self.base_ptrs[i] = (<ndarray>block).data

    def __dealloc__(self):
        free(self.base_ptrs)

    cdef move(self, int start, int end):
        cdef:
            ndarray arr
            Py_ssize_t i

        self._restore_blocks()

        # move blocks
        for i in range(self.nblocks):
            arr = self.blk_values[i]

            # axis=1 is the frame's axis=0
            arr.data = self.base_ptrs[i] + arr.strides[1] * start
            arr.shape[1] = end - start

        # move and set the index
        self.idx_slider.move(start, end)

        object.__setattr__(self.index, '_index_data', self.idx_slider.buf)
        self.index._engine.clear_mapping()
        self.index._cache.clear()  # e.g. inferred_freq must go

    cdef reset(self):
        cdef:
            ndarray arr
            Py_ssize_t i

        self._restore_blocks()

        for i in range(self.nblocks):
            arr = self.blk_values[i]

            # axis=1 is the frame's axis=0
            arr.data = self.base_ptrs[i]
            arr.shape[1] = 0

    cdef _restore_blocks(self):
        """
        Ensure that we have the original blocks, blknos, and blklocs.
        """
        mgr = self.dummy._mgr
        mgr.blocks = tuple(self.blocks)
        mgr._blklocs = self.orig_blklocs
        mgr._blknos = self.orig_blknos
