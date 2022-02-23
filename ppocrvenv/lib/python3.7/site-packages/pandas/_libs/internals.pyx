from collections import defaultdict

import cython
from cython import Py_ssize_t

from cpython.slice cimport PySlice_GetIndicesEx


cdef extern from "Python.h":
    Py_ssize_t PY_SSIZE_T_MAX

import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_INTP,
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

from pandas._libs.algos import ensure_int64

from pandas._libs.arrays cimport NDArrayBacked
from pandas._libs.util cimport is_integer_object


@cython.final
@cython.freelist(32)
cdef class BlockPlacement:
    # __slots__ = '_as_slice', '_as_array', '_len'
    cdef:
        slice _as_slice
        ndarray _as_array  # Note: this still allows `None`; will be intp_t
        bint _has_slice, _has_array, _is_known_slice_like

    def __cinit__(self, val):
        cdef:
            slice slc

        self._as_slice = None
        self._as_array = None
        self._has_slice = False
        self._has_array = False

        if is_integer_object(val):
            slc = slice(val, val + 1, 1)
            self._as_slice = slc
            self._has_slice = True
        elif isinstance(val, slice):
            slc = slice_canonize(val)

            if slc.start != slc.stop:
                self._as_slice = slc
                self._has_slice = True
            else:
                arr = np.empty(0, dtype=np.intp)
                self._as_array = arr
                self._has_array = True
        else:
            # Cython memoryview interface requires ndarray to be writeable.
            arr = np.require(val, dtype=np.intp, requirements='W')
            assert arr.ndim == 1, arr.shape
            self._as_array = arr
            self._has_array = True

    def __str__(self) -> str:
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            v = self._as_slice
        else:
            v = self._as_array

        return f"{type(self).__name__}({v})"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return slice_len(s)
        else:
            return len(self._as_array)

    def __iter__(self):
        cdef:
            slice s = self._ensure_has_slice()
            Py_ssize_t start, stop, step, _

        if s is not None:
            start, stop, step, _ = slice_get_indices_ex(s)
            return iter(range(start, stop, step))
        else:
            return iter(self._as_array)

    @property
    def as_slice(self) -> slice:
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return s
        else:
            raise TypeError("Not slice-like")

    @property
    def indexer(self):
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            return s
        else:
            return self._as_array

    @property
    def as_array(self) -> np.ndarray:
        cdef:
            Py_ssize_t start, stop, end, _

        if not self._has_array:
            start, stop, step, _ = slice_get_indices_ex(self._as_slice)
            # NOTE: this is the C-optimized equivalent of
            #  `np.arange(start, stop, step, dtype=np.intp)`
            self._as_array = cnp.PyArray_Arange(start, stop, step, NPY_INTP)
            self._has_array = True

        return self._as_array

    @property
    def is_slice_like(self) -> bool:
        cdef:
            slice s = self._ensure_has_slice()

        return s is not None

    def __getitem__(self, loc):
        cdef:
            slice s = self._ensure_has_slice()

        if s is not None:
            val = slice_getitem(s, loc)
        else:
            val = self._as_array[loc]

        if not isinstance(val, slice) and val.ndim == 0:
            return val

        return BlockPlacement(val)

    def delete(self, loc) -> BlockPlacement:
        return BlockPlacement(np.delete(self.as_array, loc, axis=0))

    def append(self, others) -> BlockPlacement:
        if not len(others):
            return self

        return BlockPlacement(
            np.concatenate([self.as_array] + [o.as_array for o in others])
        )

    cdef BlockPlacement iadd(self, other):
        cdef:
            slice s = self._ensure_has_slice()
            Py_ssize_t other_int, start, stop, step, l

        if is_integer_object(other) and s is not None:
            other_int = <Py_ssize_t>other

            if other_int == 0:
                # BlockPlacement is treated as immutable
                return self

            start, stop, step, l = slice_get_indices_ex(s)
            start += other_int
            stop += other_int

            if (step > 0 and start < 0) or (step < 0 and stop < step):
                raise ValueError("iadd causes length change")

            if stop < 0:
                val = slice(start, None, step)
            else:
                val = slice(start, stop, step)

            return BlockPlacement(val)
        else:
            newarr = self.as_array + other
            if (newarr < 0).any():
                raise ValueError("iadd causes length change")

            val = newarr
            return BlockPlacement(val)

    def add(self, other) -> BlockPlacement:
        # We can get here with int or ndarray
        return self.iadd(other)

    cdef slice _ensure_has_slice(self):
        if not self._has_slice:
            self._as_slice = indexer_as_slice(self._as_array)
            self._has_slice = True

        return self._as_slice


cdef slice slice_canonize(slice s):
    """
    Convert slice to canonical bounded form.
    """
    cdef:
        Py_ssize_t start = 0, stop = 0, step = 1

    if s.step is None:
        step = 1
    else:
        step = <Py_ssize_t>s.step
        if step == 0:
            raise ValueError("slice step cannot be zero")

    if step > 0:
        if s.stop is None:
            raise ValueError("unbounded slice")

        stop = <Py_ssize_t>s.stop
        if s.start is None:
            start = 0
        else:
            start = <Py_ssize_t>s.start
            if start > stop:
                start = stop
    elif step < 0:
        if s.start is None:
            raise ValueError("unbounded slice")

        start = <Py_ssize_t>s.start
        if s.stop is None:
            stop = -1
        else:
            stop = <Py_ssize_t>s.stop
            if stop > start:
                stop = start

    if start < 0 or (stop < 0 and s.stop is not None and step > 0):
        raise ValueError("unbounded slice")

    if stop < 0:
        return slice(start, None, step)
    else:
        return slice(start, stop, step)


cpdef Py_ssize_t slice_len(slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX) except -1:
    """
    Get length of a bounded slice.

    The slice must not have any "open" bounds that would create dependency on
    container size, i.e.:
    - if ``s.step is None or s.step > 0``, ``s.stop`` is not ``None``
    - if ``s.step < 0``, ``s.start`` is not ``None``

    Otherwise, the result is unreliable.
    """
    cdef:
        Py_ssize_t start, stop, step, length

    if slc is None:
        raise TypeError("slc must be slice")

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)

    return length


cdef slice_get_indices_ex(slice slc, Py_ssize_t objlen=PY_SSIZE_T_MAX):
    """
    Get (start, stop, step, length) tuple for a slice.

    If `objlen` is not specified, slice must be bounded, otherwise the result
    will be wrong.
    """
    cdef:
        Py_ssize_t start, stop, step, length

    if slc is None:
        raise TypeError("slc should be a slice")

    PySlice_GetIndicesEx(slc, objlen, &start, &stop, &step, &length)

    return start, stop, step, length


cdef slice_getitem(slice slc, ind):
    cdef:
        Py_ssize_t s_start, s_stop, s_step, s_len
        Py_ssize_t ind_start, ind_stop, ind_step, ind_len

    s_start, s_stop, s_step, s_len = slice_get_indices_ex(slc)

    if isinstance(ind, slice):
        ind_start, ind_stop, ind_step, ind_len = slice_get_indices_ex(ind, s_len)

        if ind_step > 0 and ind_len == s_len:
            # short-cut for no-op slice
            if ind_len == s_len:
                return slc

        if ind_step < 0:
            s_start = s_stop - s_step
            ind_step = -ind_step

        s_step *= ind_step
        s_stop = s_start + ind_stop * s_step
        s_start = s_start + ind_start * s_step

        if s_step < 0 and s_stop < 0:
            return slice(s_start, None, s_step)
        else:
            return slice(s_start, s_stop, s_step)

    else:
        # NOTE:
        # this is the C-optimized equivalent of
        # `np.arange(s_start, s_stop, s_step, dtype=np.intp)[ind]`
        return cnp.PyArray_Arange(s_start, s_stop, s_step, NPY_INTP)[ind]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef slice indexer_as_slice(intp_t[:] vals):
    cdef:
        Py_ssize_t i, n, start, stop
        int64_t d

    if vals is None:
        raise TypeError("vals must be ndarray")

    n = vals.shape[0]

    if n == 0 or vals[0] < 0:
        return None

    if n == 1:
        return slice(vals[0], vals[0] + 1, 1)

    if vals[1] < 0:
        return None

    # n > 2
    d = vals[1] - vals[0]

    if d == 0:
        return None

    for i in range(2, n):
        if vals[i] < 0 or vals[i] - vals[i - 1] != d:
            return None

    start = vals[0]
    stop = start + n * d
    if stop < 0 and d < 0:
        return slice(start, None, d)
    else:
        return slice(start, stop, d)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_blkno_indexers(
    int64_t[:] blknos, bint group=True
) -> list[tuple[int, slice | np.ndarray]]:
    """
    Enumerate contiguous runs of integers in ndarray.

    Iterate over elements of `blknos` yielding ``(blkno, slice(start, stop))``
    pairs for each contiguous run found.

    If `group` is True and there is more than one run for a certain blkno,
    ``(blkno, array)`` with an array containing positions of all elements equal
    to blkno.

    Returns
    -------
    list[tuple[int, slice | np.ndarray]]
    """
    # There's blkno in this function's name because it's used in block &
    # blockno handling.
    cdef:
        int64_t cur_blkno
        Py_ssize_t i, start, stop, n, diff, tot_len
        object blkno
        object group_dict = defaultdict(list)

    n = blknos.shape[0]
    result = list()
    start = 0
    cur_blkno = blknos[start]

    if n == 0:
        pass
    elif group is False:
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                result.append((cur_blkno, slice(start, i)))

                start = i
                cur_blkno = blknos[i]

        result.append((cur_blkno, slice(start, n)))
    else:
        for i in range(1, n):
            if blknos[i] != cur_blkno:
                group_dict[cur_blkno].append((start, i))

                start = i
                cur_blkno = blknos[i]

        group_dict[cur_blkno].append((start, n))

        for blkno, slices in group_dict.items():
            if len(slices) == 1:
                result.append((blkno, slice(slices[0][0], slices[0][1])))
            else:
                tot_len = sum(stop - start for start, stop in slices)
                arr = np.empty(tot_len, dtype=np.int64)

                i = 0
                for start, stop in slices:
                    for diff in range(start, stop):
                        arr[i] = diff
                        i += 1

                result.append((blkno, arr))

    return result


def get_blkno_placements(blknos, group: bool = True):
    """
    Parameters
    ----------
    blknos : np.ndarray[int64]
    group : bool, default True

    Returns
    -------
    iterator
        yield (blkno, BlockPlacement)
    """
    blknos = ensure_int64(blknos)

    for blkno, indexer in get_blkno_indexers(blknos, group):
        yield blkno, BlockPlacement(indexer)


@cython.freelist(64)
cdef class SharedBlock:
    """
    Defining __init__ in a cython class significantly improves performance.
    """
    cdef:
        public BlockPlacement _mgr_locs
        readonly int ndim

    def __cinit__(self, values, placement: BlockPlacement, ndim: int):
        """
        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            We assume maybe_coerce_values has already been called.
        placement : BlockPlacement
        ndim : int
            1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        """
        self._mgr_locs = placement
        self.ndim = ndim

    cpdef __reduce__(self):
        # We have to do some gymnastics b/c "ndim" is keyword-only
        from functools import partial

        from pandas.core.internals.blocks import new_block

        args = (self.values, self.mgr_locs.indexer)
        func = partial(new_block, ndim=self.ndim)
        return func, args

    cpdef __setstate__(self, state):
        from pandas.core.construction import extract_array

        self.mgr_locs = BlockPlacement(state[0])
        self.values = extract_array(state[1], extract_numpy=True)
        if len(state) > 2:
            # we stored ndim
            self.ndim = state[2]
        else:
            # older pickle
            from pandas.core.internals.api import maybe_infer_ndim

            ndim = maybe_infer_ndim(self.values, self.mgr_locs)
            self.ndim = ndim


cdef class NumpyBlock(SharedBlock):
    cdef:
        public ndarray values

    def __cinit__(self, ndarray values, BlockPlacement placement, int ndim):
        # set values here the (implicit) call to SharedBlock.__cinit__ will
        #  set placement and ndim
        self.values = values

    # @final  # not useful in cython, but we _would_ annotate with @final
    cpdef NumpyBlock getitem_block_index(self, slice slicer):
        """
        Perform __getitem__-like specialized to slicing along index.

        Assumes self.ndim == 2
        """
        new_values = self.values[..., slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim)


cdef class NDArrayBackedBlock(SharedBlock):
    """
    Block backed by NDArrayBackedExtensionArray
    """
    cdef public:
        NDArrayBacked values

    def __cinit__(self, NDArrayBacked values, BlockPlacement placement, int ndim):
        # set values here the (implicit) call to SharedBlock.__cinit__ will
        #  set placement and ndim
        self.values = values

    # @final  # not useful in cython, but we _would_ annotate with @final
    cpdef NDArrayBackedBlock getitem_block_index(self, slice slicer):
        """
        Perform __getitem__-like specialized to slicing along index.

        Assumes self.ndim == 2
        """
        new_values = self.values[..., slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim)


cdef class Block(SharedBlock):
    cdef:
        public object values

    def __cinit__(self, object values, BlockPlacement placement, int ndim):
        # set values here the (implicit) call to SharedBlock.__cinit__ will
        #  set placement and ndim
        self.values = values


@cython.freelist(64)
cdef class BlockManager:
    cdef:
        public tuple blocks
        public list axes
        public bint _known_consolidated, _is_consolidated
        public ndarray _blknos, _blklocs

    def __cinit__(self, blocks=None, axes=None, verify_integrity=True):
        # None as defaults for unpickling GH#42345
        if blocks is None:
            # This adds 1-2 microseconds to DataFrame(np.array([]))
            return

        if isinstance(blocks, list):
            # Backward compat for e.g. pyarrow
            blocks = tuple(blocks)

        self.blocks = blocks
        self.axes = axes.copy()  # copy to make sure we are not remotely-mutable

        # Populate known_consolidate, blknos, and blklocs lazily
        self._known_consolidated = False
        self._is_consolidated = False
        # error: Incompatible types in assignment (expression has type "None",
        # variable has type "ndarray")
        self._blknos = None  # type: ignore[assignment]
        # error: Incompatible types in assignment (expression has type "None",
        # variable has type "ndarray")
        self._blklocs = None  # type: ignore[assignment]

    # -------------------------------------------------------------------
    # Pickle

    cpdef __reduce__(self):
        if len(self.axes) == 1:
            # SingleBlockManager, __init__ expects Block, axis
            args = (self.blocks[0], self.axes[0])
        else:
            args = (self.blocks, self.axes)
        return type(self), args

    cpdef __setstate__(self, state):
        from pandas.core.construction import extract_array
        from pandas.core.internals.blocks import (
            ensure_block_shape,
            new_block,
        )
        from pandas.core.internals.managers import ensure_index

        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            state = state[3]["0.14.1"]
            axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(axes)

            for blk in state["blocks"]:
                vals = blk["values"]
                # older versions may hold e.g. DatetimeIndex instead of DTA
                vals = extract_array(vals, extract_numpy=True)
                blk["values"] = ensure_block_shape(vals, ndim=ndim)

            nbs = [
                new_block(blk["values"], blk["mgr_locs"], ndim=ndim)
                for blk in state["blocks"]
            ]
            blocks = tuple(nbs)
            self.blocks = blocks
            self.axes = axes

        else:
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        self._post_setstate()

    def _post_setstate(self) -> None:
        self._is_consolidated = False
        self._known_consolidated = False
        self._rebuild_blknos_and_blklocs()

    # -------------------------------------------------------------------
    # Indexing

    cdef BlockManager _get_index_slice(self, slobj):
        cdef:
            SharedBlock blk, nb

        nbs = []
        for blk in self.blocks:
            nb = blk.getitem_block_index(slobj)
            nbs.append(nb)

        new_axes = [self.axes[0], self.axes[1]._getitem_slice(slobj)]
        return type(self)(tuple(nbs), new_axes, verify_integrity=False)

    def get_slice(self, slobj: slice, axis: int = 0) -> BlockManager:

        if axis == 0:
            new_blocks = self._slice_take_blocks_ax0(slobj)
        elif axis == 1:
            return self._get_index_slice(slobj)
        else:
            raise IndexError("Requested axis not found in manager")

        new_axes = list(self.axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)

        return type(self)(tuple(new_blocks), new_axes, verify_integrity=False)
