import cython
import numpy as np

cimport numpy as cnp
from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()


# -----------------------------------------------------------------------------
# Preamble stuff

cdef float64_t NaN = <float64_t>np.NaN
cdef float64_t INF = <float64_t>np.inf

# -----------------------------------------------------------------------------


cdef class SparseIndex:
    """
    Abstract superclass for sparse index types.
    """

    def __init__(self):
        raise NotImplementedError


cdef class IntIndex(SparseIndex):
    """
    Object for holding exact integer sparse indexing information

    Parameters
    ----------
    length : integer
    indices : array-like
        Contains integers corresponding to the indices.
    check_integrity : bool, default=True
        Check integrity of the input.
    """

    cdef readonly:
        Py_ssize_t length, npoints
        ndarray indices

    def __init__(self, Py_ssize_t length, indices, bint check_integrity=True):
        self.length = length
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.npoints = len(self.indices)

        if check_integrity:
            self.check_integrity()

    def __reduce__(self):
        args = (self.length, self.indices)
        return IntIndex, args

    def __repr__(self) -> str:
        output = 'IntIndex\n'
        output += f'Indices: {repr(self.indices)}\n'
        return output

    @property
    def nbytes(self) -> int:
        return self.indices.nbytes

    def check_integrity(self):
        """
        Checks the following:

        - Indices are strictly ascending
        - Number of indices is at most self.length
        - Indices are at least 0 and at most the total length less one

        A ValueError is raised if any of these conditions is violated.
        """

        if self.npoints > self.length:
            raise ValueError(
                f"Too many indices. Expected {self.length} but found {self.npoints}"
            )

        # Indices are vacuously ordered and non-negative
        # if the sequence of indices is empty.
        if self.npoints == 0:
            return

        if self.indices.min() < 0:
            raise ValueError("No index can be less than zero")

        if self.indices.max() >= self.length:
            raise ValueError("All indices must be less than the length")

        monotonic = np.all(self.indices[:-1] < self.indices[1:])
        if not monotonic:
            raise ValueError("Indices must be strictly increasing")

    def equals(self, other: object) -> bool:
        if not isinstance(other, IntIndex):
            return False

        if self is other:
            return True

        same_length = self.length == other.length
        same_indices = np.array_equal(self.indices, other.indices)
        return same_length and same_indices

    @property
    def ngaps(self) -> int:
        return self.length - self.npoints

    def to_int_index(self):
        return self

    def to_block_index(self):
        locs, lens = get_blocks(self.indices)
        return BlockIndex(self.length, locs, lens)

    cpdef IntIndex intersect(self, SparseIndex y_):
        cdef:
            Py_ssize_t out_length, xi, yi = 0, result_indexer = 0
            int32_t xind
            ndarray[int32_t, ndim=1] xindices, yindices, new_indices
            IntIndex y

        # if is one already, returns self
        y = y_.to_int_index()

        if self.length != y.length:
            raise Exception('Indices must reference same underlying length')

        xindices = self.indices
        yindices = y.indices
        new_indices = np.empty(min(
            len(xindices), len(yindices)), dtype=np.int32)

        for xi in range(self.npoints):
            xind = xindices[xi]

            while yi < y.npoints and yindices[yi] < xind:
                yi += 1

            if yi >= y.npoints:
                break

            # TODO: would a two-pass algorithm be faster?
            if yindices[yi] == xind:
                new_indices[result_indexer] = xind
                result_indexer += 1

        new_indices = new_indices[:result_indexer]
        return IntIndex(self.length, new_indices)

    cpdef IntIndex make_union(self, SparseIndex y_):

        cdef:
            ndarray[int32_t, ndim=1] new_indices
            IntIndex y

        # if is one already, returns self
        y = y_.to_int_index()

        if self.length != y.length:
            raise ValueError('Indices must reference same underlying length')

        new_indices = np.union1d(self.indices, y.indices)
        return IntIndex(self.length, new_indices)

    @cython.wraparound(False)
    cpdef int32_t lookup(self, Py_ssize_t index):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            int32_t res
            ndarray[int32_t, ndim=1] inds

        inds = self.indices
        if self.npoints == 0:
            return -1
        elif index < 0 or self.length <= index:
            return -1

        res = inds.searchsorted(index)
        if res == self.npoints:
            return -1
        elif inds[res] == index:
            return res
        else:
            return -1

    @cython.wraparound(False)
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        cdef:
            Py_ssize_t n, i, ind_val
            ndarray[int32_t, ndim=1] inds
            ndarray[uint8_t, ndim=1, cast=True] mask
            ndarray[int32_t, ndim=1] masked
            ndarray[int32_t, ndim=1] res
            ndarray[int32_t, ndim=1] results

        n = len(indexer)
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        if self.npoints == 0:
            return results

        inds = self.indices
        mask = (inds[0] <= indexer) & (indexer <= inds[len(inds) - 1])

        masked = indexer[mask]
        res = inds.searchsorted(masked).astype(np.int32)

        res[inds[res] != masked] = -1
        results[mask] = res
        return results

    cpdef ndarray reindex(self, ndarray[float64_t, ndim=1] values,
                          float64_t fill_value, SparseIndex other_):
        cdef:
            Py_ssize_t i = 0, j = 0
            IntIndex other
            ndarray[float64_t, ndim=1] result
            ndarray[int32_t, ndim=1] sinds, oinds

        other = other_.to_int_index()

        oinds = other.indices
        sinds = self.indices

        result = np.empty(other.npoints, dtype=np.float64)
        result[:] = fill_value

        for i in range(other.npoints):
            while oinds[i] > sinds[j] and j < self.npoints:
                j += 1

            if j == self.npoints:
                break

            if oinds[i] < sinds[j]:
                continue
            elif oinds[i] == sinds[j]:
                result[i] = values[j]
                j += 1

        return result

    cpdef put(self, ndarray[float64_t, ndim=1] values,
              ndarray[int32_t, ndim=1] indices, object to_put):
        pass

    cpdef take(self, ndarray[float64_t, ndim=1] values,
               ndarray[int32_t, ndim=1] indices):
        pass


cpdef get_blocks(ndarray[int32_t, ndim=1] indices):
    cdef:
        Py_ssize_t init_len, i, npoints, result_indexer = 0
        int32_t block, length = 1, cur, prev
        ndarray[int32_t, ndim=1] locs, lens

    npoints = len(indices)

    # just handle the special empty case separately
    if npoints == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # block size can't be longer than npoints
    locs = np.empty(npoints, dtype=np.int32)
    lens = np.empty(npoints, dtype=np.int32)

    # TODO: two-pass algorithm faster?
    prev = block = indices[0]
    for i in range(1, npoints):
        cur = indices[i]
        if cur - prev > 1:
            # new block
            locs[result_indexer] = block
            lens[result_indexer] = length
            block = cur
            length = 1
            result_indexer += 1
        else:
            # same block, increment length
            length += 1

        prev = cur

    locs[result_indexer] = block
    lens[result_indexer] = length
    result_indexer += 1
    locs = locs[:result_indexer]
    lens = lens[:result_indexer]
    return locs, lens


# -----------------------------------------------------------------------------
# BlockIndex

cdef class BlockIndex(SparseIndex):
    """
    Object for holding block-based sparse indexing information

    Parameters
    ----------
    """
    cdef readonly:
        int32_t nblocks, npoints, length
        ndarray blocs, blengths

    cdef:
        object __weakref__  # need to be picklable
        int32_t *locbuf
        int32_t *lenbuf

    def __init__(self, length, blocs, blengths):

        self.blocs = np.ascontiguousarray(blocs, dtype=np.int32)
        self.blengths = np.ascontiguousarray(blengths, dtype=np.int32)

        # in case we need
        self.locbuf = <int32_t*>self.blocs.data
        self.lenbuf = <int32_t*>self.blengths.data

        self.length = length
        self.nblocks = np.int32(len(self.blocs))
        self.npoints = self.blengths.sum()

        # self.block_start = blocs
        # self.block_end = blocs + blengths

        self.check_integrity()

    def __reduce__(self):
        args = (self.length, self.blocs, self.blengths)
        return BlockIndex, args

    def __repr__(self) -> str:
        output = 'BlockIndex\n'
        output += f'Block locations: {repr(self.blocs)}\n'
        output += f'Block lengths: {repr(self.blengths)}'

        return output

    @property
    def nbytes(self) -> int:
        return self.blocs.nbytes + self.blengths.nbytes

    @property
    def ngaps(self) -> int:
        return self.length - self.npoints

    cpdef check_integrity(self):
        """
        Check:
        - Locations are in ascending order
        - No overlapping blocks
        - Blocks to not start after end of index, nor extend beyond end
        """
        cdef:
            Py_ssize_t i
            ndarray[int32_t, ndim=1] blocs, blengths

        blocs = self.blocs
        blengths = self.blengths

        if len(blocs) != len(blengths):
            raise ValueError('block bound arrays must be same length')

        for i in range(self.nblocks):
            if i > 0:
                if blocs[i] <= blocs[i - 1]:
                    raise ValueError('Locations not in ascending order')

            if i < self.nblocks - 1:
                if blocs[i] + blengths[i] > blocs[i + 1]:
                    raise ValueError(f'Block {i} overlaps')
            else:
                if blocs[i] + blengths[i] > self.length:
                    raise ValueError(f'Block {i} extends beyond end')

            # no zero-length blocks
            if blengths[i] == 0:
                raise ValueError(f'Zero-length block {i}')

    def equals(self, other: object) -> bool:
        if not isinstance(other, BlockIndex):
            return False

        if self is other:
            return True

        same_length = self.length == other.length
        same_blocks = (np.array_equal(self.blocs, other.blocs) and
                       np.array_equal(self.blengths, other.blengths))
        return same_length and same_blocks

    def to_block_index(self):
        return self

    def to_int_index(self):
        cdef:
            int32_t i = 0, j, b
            int32_t offset
            ndarray[int32_t, ndim=1] indices

        indices = np.empty(self.npoints, dtype=np.int32)

        for b in range(self.nblocks):
            offset = self.locbuf[b]

            for j in range(self.lenbuf[b]):
                indices[i] = offset + j
                i += 1

        return IntIndex(self.length, indices)

    cpdef BlockIndex intersect(self, SparseIndex other):
        """
        Intersect two BlockIndex objects

        Returns
        -------
        BlockIndex
        """
        cdef:
            BlockIndex y
            ndarray[int32_t, ndim=1] xloc, xlen, yloc, ylen, out_bloc, out_blen
            Py_ssize_t xi = 0, yi = 0, max_len, result_indexer = 0
            int32_t cur_loc, cur_length, diff

        y = other.to_block_index()

        if self.length != y.length:
            raise Exception('Indices must reference same underlying length')

        xloc = self.blocs
        xlen = self.blengths
        yloc = y.blocs
        ylen = y.blengths

        # block may be split, but can't exceed original len / 2 + 1
        max_len = min(self.length, y.length) // 2 + 1
        out_bloc = np.empty(max_len, dtype=np.int32)
        out_blen = np.empty(max_len, dtype=np.int32)

        while True:
            # we are done (or possibly never began)
            if xi >= self.nblocks or yi >= y.nblocks:
                break

            # completely symmetric...would like to avoid code dup but oh well
            if xloc[xi] >= yloc[yi]:
                cur_loc = xloc[xi]
                diff = xloc[xi] - yloc[yi]

                if ylen[yi] <= diff:
                    # have to skip this block
                    yi += 1
                    continue

                if ylen[yi] - diff < xlen[xi]:
                    # take end of y block, move onward
                    cur_length = ylen[yi] - diff
                    yi += 1
                else:
                    # take end of x block
                    cur_length = xlen[xi]
                    xi += 1

            else:  # xloc[xi] < yloc[yi]
                cur_loc = yloc[yi]
                diff = yloc[yi] - xloc[xi]

                if xlen[xi] <= diff:
                    # have to skip this block
                    xi += 1
                    continue

                if xlen[xi] - diff < ylen[yi]:
                    # take end of x block, move onward
                    cur_length = xlen[xi] - diff
                    xi += 1
                else:
                    # take end of y block
                    cur_length = ylen[yi]
                    yi += 1

            out_bloc[result_indexer] = cur_loc
            out_blen[result_indexer] = cur_length
            result_indexer += 1

        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        return BlockIndex(self.length, out_bloc, out_blen)

    cpdef BlockIndex make_union(self, SparseIndex y):
        """
        Combine together two BlockIndex objects, accepting indices if contained
        in one or the other

        Parameters
        ----------
        other : SparseIndex

        Notes
        -----
        union is a protected keyword in Cython, hence make_union

        Returns
        -------
        BlockIndex
        """
        return BlockUnion(self, y.to_block_index()).result

    cpdef Py_ssize_t lookup(self, Py_ssize_t index):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
        cdef:
            Py_ssize_t i, cum_len
            ndarray[int32_t, ndim=1] locs, lens

        locs = self.blocs
        lens = self.blengths

        if self.nblocks == 0:
            return -1
        elif index < locs[0]:
            return -1

        cum_len = 0
        for i in range(self.nblocks):
            if index >= locs[i] and index < locs[i] + lens[i]:
                return cum_len + index - locs[i]
            cum_len += lens[i]

        return -1

    @cython.wraparound(False)
    cpdef ndarray[int32_t] lookup_array(self, ndarray[int32_t, ndim=1] indexer):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
        cdef:
            Py_ssize_t n, i, j, ind_val
            ndarray[int32_t, ndim=1] locs, lens
            ndarray[int32_t, ndim=1] results

        locs = self.blocs
        lens = self.blengths

        n = len(indexer)
        results = np.empty(n, dtype=np.int32)
        results[:] = -1

        if self.npoints == 0:
            return results

        for i in range(n):
            ind_val = indexer[i]
            if not (ind_val < 0 or self.length <= ind_val):
                cum_len = 0
                for j in range(self.nblocks):
                    if ind_val >= locs[j] and ind_val < locs[j] + lens[j]:
                        results[i] = cum_len + ind_val - locs[j]
                    cum_len += lens[j]
        return results

    cpdef ndarray reindex(self, ndarray[float64_t, ndim=1] values,
                          float64_t fill_value, SparseIndex other_):
        cdef:
            Py_ssize_t i = 0, j = 0, ocur, ocurlen
            BlockIndex other
            ndarray[float64_t, ndim=1] result
            ndarray[int32_t, ndim=1] slocs, slens, olocs, olens

        other = other_.to_block_index()

        olocs = other.blocs
        olens = other.blengths
        slocs = self.blocs
        slens = self.blengths

        result = np.empty(other.npoints, dtype=np.float64)

        for i in range(other.nblocks):
            ocur = olocs[i]
            ocurlen = olens[i]

            while slocs[j] + slens[j] < ocur:
                j += 1

    cpdef put(self, ndarray[float64_t, ndim=1] values,
              ndarray[int32_t, ndim=1] indices, object to_put):
        pass

    cpdef take(self, ndarray[float64_t, ndim=1] values,
               ndarray[int32_t, ndim=1] indices):
        pass


@cython.internal
cdef class BlockMerge:
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """
    cdef:
        BlockIndex x, y, result
        ndarray xstart, xlen, xend, ystart, ylen, yend
        int32_t xi, yi  # block indices

    def __init__(self, BlockIndex x, BlockIndex y):
        self.x = x
        self.y = y

        if x.length != y.length:
            raise Exception('Indices must reference same underlying length')

        self.xstart = self.x.blocs
        self.ystart = self.y.blocs

        self.xend = self.x.blocs + self.x.blengths
        self.yend = self.y.blocs + self.y.blengths

        # self.xlen = self.x.blengths
        # self.ylen = self.y.blengths

        self.xi = 0
        self.yi = 0

        self.result = self._make_merged_blocks()

    cdef _make_merged_blocks(self):
        raise NotImplementedError

    cdef _set_current_indices(self, int32_t xi, int32_t yi, bint mode):
        if mode == 0:
            self.xi = xi
            self.yi = yi
        else:
            self.xi = yi
            self.yi = xi


@cython.internal
cdef class BlockUnion(BlockMerge):
    """
    Object-oriented approach makes sharing state between recursive functions a
    lot easier and reduces code duplication
    """

    cdef _make_merged_blocks(self):
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart
            ndarray[int32_t, ndim=1] yend, out_bloc, out_blen
            int32_t nstart, nend, diff
            Py_ssize_t max_len, result_indexer = 0

        xstart = self.xstart
        xend = self.xend
        ystart = self.ystart
        yend = self.yend

        max_len = min(self.x.length, self.y.length) // 2 + 1
        out_bloc = np.empty(max_len, dtype=np.int32)
        out_blen = np.empty(max_len, dtype=np.int32)

        while True:
            # we are done (or possibly never began)
            if self.xi >= self.x.nblocks and self.yi >= self.y.nblocks:
                break
            elif self.yi >= self.y.nblocks:
                # through with y, just pass through x blocks
                nstart = xstart[self.xi]
                nend = xend[self.xi]
                self.xi += 1
            elif self.xi >= self.x.nblocks:
                # through with x, just pass through y blocks
                nstart = ystart[self.yi]
                nend = yend[self.yi]
                self.yi += 1
            else:
                # find end of new block
                if xstart[self.xi] < ystart[self.yi]:
                    nstart = xstart[self.xi]
                    nend = self._find_next_block_end(0)
                else:
                    nstart = ystart[self.yi]
                    nend = self._find_next_block_end(1)

            out_bloc[result_indexer] = nstart
            out_blen[result_indexer] = nend - nstart
            result_indexer += 1

        out_bloc = out_bloc[:result_indexer]
        out_blen = out_blen[:result_indexer]

        return BlockIndex(self.x.length, out_bloc, out_blen)

    cdef int32_t _find_next_block_end(self, bint mode) except -1:
        """
        Wow, this got complicated in a hurry

        mode 0: block started in index x
        mode 1: block started in index y
        """
        cdef:
            ndarray[int32_t, ndim=1] xstart, xend, ystart, yend
            int32_t xi, yi, xnblocks, ynblocks, nend

        if mode != 0 and mode != 1:
            raise Exception('Mode must be 0 or 1')

        # so symmetric code will work
        if mode == 0:
            xstart = self.xstart
            xend = self.xend
            xi = self.xi

            ystart = self.ystart
            yend = self.yend
            yi = self.yi
            ynblocks = self.y.nblocks
        else:
            xstart = self.ystart
            xend = self.yend
            xi = self.yi

            ystart = self.xstart
            yend = self.xend
            yi = self.xi
            ynblocks = self.x.nblocks

        nend = xend[xi]

        # done with y?
        if yi == ynblocks:
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        elif nend < ystart[yi]:
            # block ends before y block
            self._set_current_indices(xi + 1, yi, mode)
            return nend
        else:
            while yi < ynblocks and nend > yend[yi]:
                yi += 1

            self._set_current_indices(xi + 1, yi, mode)

            if yi == ynblocks:
                return nend

            if nend < ystart[yi]:
                # we're done, return the block end
                return nend
            else:
                # merge blocks, continue searching
                # this also catches the case where blocks
                return self._find_next_block_end(1 - mode)


# -----------------------------------------------------------------------------
# Sparse arithmetic

include "sparse_op_helper.pxi"


# -----------------------------------------------------------------------------
# SparseArray mask create operations

def make_mask_object_ndarray(ndarray[object, ndim=1] arr, object fill_value):
    cdef:
        object value
        Py_ssize_t i
        Py_ssize_t new_length = len(arr)
        ndarray[int8_t, ndim=1] mask

    mask = np.ones(new_length, dtype=np.int8)

    for i in range(new_length):
        value = arr[i]
        if value == fill_value and type(value) == type(fill_value):
            mask[i] = 0

    return mask.view(dtype=bool)
