# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2020 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements support for high-level dataset access.
"""

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property
import posixpath as pp
import sys
from warnings import warn

from threading import local

import numpy

from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from ..h5py_warnings import H5pyDeprecationWarning
from .base import HLObject, phil, with_phil, Empty, find_item_type
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support

_LEGACY_GZIP_COMPRESSION_VALS = frozenset(range(10))
MPI = h5.get_config().mpi


def make_new_dset(parent, shape=None, dtype=None, data=None, name=None,
                  chunks=None, compression=None, shuffle=None,
                  fletcher32=None, maxshape=None, compression_opts=None,
                  fillvalue=None, scaleoffset=None, track_times=False,
                  external=None, track_order=None, dcpl=None,
                  allow_unknown_filter=False):
    """ Return a new low-level dataset identifier """

    # Convert data to a C-contiguous ndarray
    if data is not None and not isinstance(data, Empty):
        from . import base
        data = base.array_for_new_object(data, specified_dtype=dtype)

    # Validate shape
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified")
            data = Empty(dtype)
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (numpy.product(shape, dtype=numpy.ulonglong) != numpy.product(data.shape, dtype=numpy.ulonglong)):
            raise ValueError("Shape tuple is incompatible with data")

    if isinstance(maxshape, int):
        maxshape = (maxshape,)
    tmp_shape = maxshape if maxshape is not None else shape

    # Validate chunk shape
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)
    if isinstance(chunks, tuple) and any(
        chunk > dim for dim, chunk in zip(tmp_shape, chunks) if dim is not None
    ):
        errmsg = "Chunk shape must not be greater than data shape in any dimension. "\
                 "{} is not compatible with {}".format(chunks, shape)
        raise ValueError(errmsg)

    if isinstance(dtype, Datatype):
        # Named types are used as-is
        tid = dtype.id
        dtype = tid.dtype  # Following code needs this
    else:
        # Validate dtype
        if dtype is None and data is None:
            dtype = numpy.dtype("=f4")
        elif dtype is None and data is not None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)
        tid = h5t.py_create(dtype, logical=1)

    # Legacy
    if any((compression, shuffle, fletcher32, maxshape, scaleoffset)) and chunks is False:
        raise ValueError("Chunked format required for given storage options")

    # Legacy
    if compression is True:
        if compression_opts is None:
            compression_opts = 4
        compression = 'gzip'

    # Legacy
    if compression in _LEGACY_GZIP_COMPRESSION_VALS:
        if compression_opts is not None:
            raise TypeError("Conflict in compression options")
        compression_opts = compression
        compression = 'gzip'
    dcpl = filters.fill_dcpl(
        dcpl or h5p.create(h5p.DATASET_CREATE), shape, dtype,
        chunks, compression, compression_opts, shuffle, fletcher32,
        maxshape, scaleoffset, external, allow_unknown_filter)

    if fillvalue is not None:
        fillvalue = numpy.array(fillvalue)
        dcpl.set_fill_value(fillvalue)

    if track_times is None:
        # In case someone explicitly passes None for the default
        track_times = False
    if track_times in (True, False):
        dcpl.set_obj_track_times(track_times)
    else:
        raise TypeError("track_times must be either True or False")
    if track_order is True:
        dcpl.set_attr_creation_order(
            h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    elif track_order is False:
        dcpl.set_attr_creation_order(0)
    elif track_order is not None:
        raise TypeError("track_order must be either True or False")

    if maxshape is not None:
        maxshape = tuple(m if m is not None else h5s.UNLIMITED for m in maxshape)

    if isinstance(data, Empty):
        sid = h5s.create(h5s.NULL)
    else:
        sid = h5s.create_simple(shape, maxshape)


    dset_id = h5d.create(parent.id, name, tid, sid, dcpl=dcpl)

    if (data is not None) and (not isinstance(data, Empty)):
        dset_id.write(h5s.ALL, h5s.ALL, data)

    return dset_id


class AstypeWrapper(object):
    """Wrapper to convert data on reading from a dataset.
    """
    def __init__(self, dset, dtype):
        self._dset = dset
        self._dtype = numpy.dtype(dtype)

    def __getitem__(self, args):
        return self._dset.__getitem__(args, new_dtype=self._dtype)

    def __enter__(self):
        # pylint: disable=protected-access
        warn(
            "Using astype() as a context manager is deprecated. "
            "Slice the returned object instead, like: ds.astype(np.int32)[:10]",
            category=H5pyDeprecationWarning, stacklevel=2,
        )
        self._dset._local.astype = self._dtype
        return self

    def __exit__(self, *args):
        # pylint: disable=protected-access
        self._dset._local.astype = None

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.astype('f8'))
        """
        return len(self._dset)


class AsStrWrapper:
    """Wrapper to decode strings on reading the dataset"""
    def __init__(self, dset, encoding, errors='strict'):
        self._dset = dset
        if encoding is None:
            encoding = h5t.check_string_dtype(dset.dtype).encoding
        self.encoding = encoding
        self.errors = errors

    def __getitem__(self, args):
        bytes_arr = self._dset[args]
        # numpy.char.decode() seems like the obvious thing to use. But it only
        # accepts numpy string arrays, not object arrays of bytes (which we
        # return from HDF5 variable-length strings). And the numpy
        # implementation is not faster than doing it with a loop; in fact, by
        # not converting the result to a numpy unicode array, the
        # naive way can be faster! (Comparing with numpy 1.18.4, June 2020)
        if numpy.isscalar(bytes_arr):
            return bytes_arr.decode(self.encoding, self.errors)

        return numpy.array([
            b.decode(self.encoding, self.errors) for b in bytes_arr.flat
        ], dtype=object).reshape(bytes_arr.shape)

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.asstr())
        """
        return len(self._dset)


class FieldsWrapper:
    """Wrapper to extract named fields from a dataset with a struct dtype"""
    extract_field = None

    def __init__(self, dset, prior_dtype, names):
        self._dset = dset
        if isinstance(names, str):
            self.extract_field = names
            names = [names]
        self.read_dtype = readtime_dtype(prior_dtype, names)

    def __getitem__(self, args):
        data = self._dset.__getitem__(args, new_dtype=self.read_dtype)
        if self.extract_field is not None:
            data = data[self.extract_field]
        return data

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.fields(['x', 'y']))
        """
        return len(self._dset)


def readtime_dtype(basetype, names):
    """Make a NumPy compound dtype with a subset of available fields"""
    if basetype.names is None:  # Names provided, but not compound
        raise ValueError("Field names only allowed for compound types")

    for name in names:  # Check all names are legal
        if not name in basetype.names:
            raise ValueError("Field %s does not appear in this type." % name)

    return numpy.dtype([(name, basetype.fields[name][0]) for name in names])


if MPI:
    class CollectiveContext(object):

        """ Manages collective I/O in MPI mode """

        # We don't bother with _local as threads are forbidden in MPI mode

        def __init__(self, dset):
            self._dset = dset

        def __enter__(self):
            # pylint: disable=protected-access
            self._dset._dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

        def __exit__(self, *args):
            # pylint: disable=protected-access
            self._dset._dxpl.set_dxpl_mpio(h5fd.MPIO_INDEPENDENT)


class ChunkIterator(object):
    """
    Class to iterate through list of chunks of a given dataset
    """
    def __init__(self, dset, source_sel=None):
        self._shape = dset.shape
        rank = len(dset.shape)

        if not dset.chunks:
            # can only use with chunked datasets
            raise TypeError("Chunked dataset required")

        self._layout = dset.chunks
        if source_sel is None:
            # select over entire dataset
            slices = []
            for dim in range(rank):
                slices.append(slice(0, self._shape[dim]))
            self._sel = tuple(slices)
        else:
            if isinstance(source_sel, slice):
                self._sel = (source_sel,)
            else:
                self._sel = source_sel
        if len(self._sel) != rank:
            raise ValueError("Invalid selection - selection region must have same rank as dataset")
        self._chunk_index = []
        for dim in range(rank):
            s = self._sel[dim]
            if s.start < 0 or s.stop > self._shape[dim] or s.stop <= s.start:
                raise ValueError("Invalid selection - selection region must be within dataset space")
            index = s.start // self._layout[dim]
            self._chunk_index.append(index)

    def __iter__(self):
        return self

    def __next__(self):
        rank = len(self._shape)
        slices = []
        if rank == 0 or self._chunk_index[0] * self._layout[0] >= self._sel[0].stop:
            # ran past the last chunk, end iteration
            raise StopIteration()

        for dim in range(rank):
            s = self._sel[dim]
            start = self._chunk_index[dim] * self._layout[dim]
            stop = (self._chunk_index[dim] + 1) * self._layout[dim]
            # adjust the start if this is an edge chunk
            if start < s.start:
                start = s.start
            if stop > s.stop:
                stop = s.stop  # trim to end of the selection
            s = slice(start, stop, 1)
            slices.append(s)

        # bump up the last index and carry forward if we run outside the selection
        dim = rank - 1
        while dim >= 0:
            s = self._sel[dim]
            self._chunk_index[dim] += 1

            chunk_end = self._chunk_index[dim] * self._layout[dim]
            if chunk_end < s.stop:
                # we still have room to extend along this dimensions
                return tuple(slices)

            if dim > 0:
                # reset to the start and continue iterating with higher dimension
                self._chunk_index[dim] = 0
            dim -= 1
        return tuple(slices)

class Dataset(HLObject):

    """
        Represents an HDF5 dataset
    """

    def astype(self, dtype):
        """ Get a wrapper allowing you to perform reads to a
        different destination type, e.g.:

        >>> double_precision = dataset.astype('f8')[0:100:2]
        """
        return AstypeWrapper(self, dtype)

    def asstr(self, encoding=None, errors='strict'):
        """Get a wrapper to read string data as Python strings:

        >>> str_array = dataset.asstr()[:]

        The parameters have the same meaning as in ``bytes.decode()``.
        If ``encoding`` is unspecified, it will use the encoding in the HDF5
        datatype (either ascii or utf-8).
        """
        string_info = h5t.check_string_dtype(self.dtype)
        if string_info is None:
            raise TypeError(
                "dset.asstr() can only be used on datasets with "
                "an HDF5 string datatype"
            )
        if encoding is None:
            encoding = string_info.encoding
        return AsStrWrapper(self, encoding, errors=errors)

    def fields(self, names, *, _prior_dtype=None):
        """Get a wrapper to read a subset of fields from a compound data type:

        >>> 2d_coords = dataset.fields(['x', 'y'])[:]

        If names is a string, a single field is extracted, and the resulting
        arrays will have that dtype. Otherwise, it should be an iterable,
        and the read data will have a compound dtype.
        """
        if _prior_dtype is None:
            _prior_dtype = self.dtype
        return FieldsWrapper(self, _prior_dtype, names)

    if MPI:
        @property
        @with_phil
        def collective(self):
            """ Context manager for MPI collective reads & writes """
            return CollectiveContext(self)

    @property
    def dims(self):
        """ Access dimension scales attached to this dataset. """
        from .dims import DimensionManager
        with phil:
            return DimensionManager(self)

    @property
    @with_phil
    def ndim(self):
        """Numpy-style attribute giving the number of dimensions"""
        return self.id.rank

    @property
    def shape(self):
        """Numpy-style shape tuple giving dataset dimensions"""
        if 'shape' in self._cache_props:
            return self._cache_props['shape']

        with phil:
            shape = self.id.shape

        # If the file is read-only, cache the shape to speed-up future uses.
        # This cache is invalidated by .refresh() when using SWMR.
        if self._readonly:
            self._cache_props['shape'] = shape
        return shape

    @shape.setter
    @with_phil
    def shape(self, shape):
        # pylint: disable=missing-docstring
        self.resize(shape)

    @property
    def size(self):
        """Numpy-style attribute giving the total dataset size"""
        if 'size' in self._cache_props:
            return self._cache_props['size']

        if self._is_empty:
            size = None
        else:
            size = numpy.prod(self.shape, dtype=numpy.intp)

        # If the file is read-only, cache the size to speed-up future uses.
        # This cache is invalidated by .refresh() when using SWMR.
        if self._readonly:
            self._cache_props['size'] = size
        return size

    @property
    def nbytes(self):
        """Numpy-style attribute giving the raw dataset size as the number of bytes"""
        size = self.size
        if size is None:  # if we are an empty 0-D array, then there are no bytes in the dataset
            return 0
        return self.dtype.itemsize * size

    @property
    def _selector(self):
        """Internal object for optimised selection of data"""
        if '_selector' in self._cache_props:
            return self._cache_props['_selector']

        slr = _selector.Selector(self.id.get_space())

        # If the file is read-only, cache the reader to speed up future uses.
        # This cache is invalidated by .refresh() when using SWMR.
        if self._readonly:
            self._cache_props['_selector'] = slr
        return slr

    @property
    def _fast_reader(self):
        """Internal object for optimised reading of data"""
        if '_fast_reader' in self._cache_props:
            return self._cache_props['_fast_reader']

        rdr = _selector.Reader(self.id)

        # If the file is read-only, cache the reader to speed up future uses.
        # This cache is invalidated by .refresh() when using SWMR.
        if self._readonly:
            self._cache_props['_fast_reader'] = rdr
        return rdr

    @property
    @with_phil
    def dtype(self):
        """Numpy dtype representing the datatype"""
        return self.id.dtype

    @property
    @with_phil
    def chunks(self):
        """Dataset chunks (or None)"""
        dcpl = self._dcpl
        if dcpl.get_layout() == h5d.CHUNKED:
            return dcpl.get_chunk()
        return None

    @property
    @with_phil
    def compression(self):
        """Compression strategy (or None)"""
        for x in ('gzip','lzf','szip'):
            if x in self._filters:
                return x
        return None

    @property
    @with_phil
    def compression_opts(self):
        """ Compression setting.  Int(0-9) for gzip, 2-tuple for szip. """
        return self._filters.get(self.compression, None)

    @property
    @with_phil
    def shuffle(self):
        """Shuffle filter present (T/F)"""
        return 'shuffle' in self._filters

    @property
    @with_phil
    def fletcher32(self):
        """Fletcher32 filter is present (T/F)"""
        return 'fletcher32' in self._filters

    @property
    @with_phil
    def scaleoffset(self):
        """Scale/offset filter settings. For integer data types, this is
        the number of bits stored, or 0 for auto-detected. For floating
        point data types, this is the number of decimal places retained.
        If the scale/offset filter is not in use, this is None."""
        try:
            return self._filters['scaleoffset'][1]
        except KeyError:
            return None

    @property
    @with_phil
    def external(self):
        """External file settings. Returns a list of tuples of
        (name, offset, size) for each external file entry, or returns None
        if no external files are used."""
        count = self._dcpl.get_external_count()
        if count<=0:
            return None
        ext_list = list()
        for x in range(count):
            (name, offset, size) = self._dcpl.get_external(x)
            ext_list.append( (filename_decode(name), offset, size) )
        return ext_list

    @property
    @with_phil
    def maxshape(self):
        """Shape up to which this dataset can be resized.  Axes with value
        None have no resize limit. """
        space = self.id.get_space()
        dims = space.get_simple_extent_dims(True)
        if dims is None:
            return None

        return tuple(x if x != h5s.UNLIMITED else None for x in dims)

    @property
    @with_phil
    def fillvalue(self):
        """Fill value for this dataset (0 by default)"""
        arr = numpy.ndarray((1,), dtype=self.dtype)
        self._dcpl.get_fill_value(arr)
        return arr[0]

    @cached_property
    @with_phil
    def _extent_type(self):
        """Get extent type for this dataset - SIMPLE, SCALAR or NULL"""
        return self.id.get_space().get_simple_extent_type()

    @cached_property
    def _is_empty(self):
        """Check if extent type is empty"""
        return self._extent_type == h5s.NULL

    @with_phil
    def __init__(self, bind, *, readonly=False):
        """ Create a new Dataset object by binding to a low-level DatasetID.
        """
        if not isinstance(bind, h5d.DatasetID):
            raise ValueError("%s is not a DatasetID" % bind)
        super(Dataset, self).__init__(bind)

        self._dcpl = self.id.get_create_plist()
        self._dxpl = h5p.create(h5p.DATASET_XFER)
        self._filters = filters.get_filters(self._dcpl)
        self._readonly = readonly
        self._cache_props = {}
        self._local = local()
        self._local.astype = None

    def resize(self, size, axis=None):
        """ Resize the dataset, or the specified axis.

        The dataset must be stored in chunked format; it can be resized up to
        the "maximum shape" (keyword maxshape) specified at creation time.
        The rank of the dataset cannot be changed.

        "Size" should be a shape tuple, or if an axis is specified, an integer.

        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data are
        fixed.
        """
        with phil:
            if self.chunks is None:
                raise TypeError("Only chunked datasets can be resized")

            if axis is not None:
                if not (axis >=0 and axis < self.id.rank):
                    raise ValueError("Invalid axis (0 to %s allowed)" % (self.id.rank-1))
                try:
                    newlen = int(size)
                except TypeError:
                    raise TypeError("Argument must be a single int if axis is specified")
                size = list(self.shape)
                size[axis] = newlen

            size = tuple(size)
            self.id.set_extent(size)
            #h5f.flush(self.id)  # THG recommends

    @with_phil
    def __len__(self):
        """ The size of the first axis.  TypeError if scalar.

        Limited to 2**32 on 32-bit systems; Dataset.len() is preferred.
        """
        size = self.len()
        if size > sys.maxsize:
            raise OverflowError("Value too big for Python's __len__; use Dataset.len() instead.")
        return size

    def len(self):
        """ The size of the first axis.  TypeError if scalar.

        Use of this method is preferred to len(dset), as Python's built-in
        len() cannot handle values greater then 2**32 on 32-bit systems.
        """
        with phil:
            shape = self.shape
            if len(shape) == 0:
                raise TypeError("Attempt to take len() of scalar dataset")
            return shape[0]

    @with_phil
    def __iter__(self):
        """ Iterate over the first axis.  TypeError if scalar.

        BEWARE: Modifications to the yielded data are *NOT* written to file.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    @with_phil
    def iter_chunks(self, sel=None):
        """ Return chunk iterator.  If set, the sel argument is a slice or
        tuple of slices that defines the region to be used. If not set, the
        entire dataspace will be used for the iterator.

        For each chunk within the given region, the iterator yields a tuple of
        slices that gives the intersection of the given chunk with the
        selection area.

        A TypeError will be raised if the dataset is not chunked.

        A ValueError will be raised if the selection region is invalid.

        """
        return ChunkIterator(self, sel)

    @cached_property
    def _fast_read_ok(self):
        """Is this dataset suitable for simple reading"""
        return (
            self._extent_type == h5s.SIMPLE
            and isinstance(self.id.get_type(), (h5t.TypeIntegerID, h5t.TypeFloatID))
        )

    @with_phil
    def __getitem__(self, args, new_dtype=None):
        """ Read a slice from the HDF5 dataset.

        Takes slices and recarray-style field names (more than one is
        allowed!) in any order.  Obeys basic NumPy rules, including
        broadcasting.

        Also supports:

        * Boolean "mask" array indexing
        """
        args = args if isinstance(args, tuple) else (args,)

        if new_dtype is None:
            new_dtype = getattr(self._local, 'astype', None)

        if self._fast_read_ok and (new_dtype is None):
            try:
                return self._fast_reader.read(args)
            except TypeError:
                pass  # Fall back to Python read pathway below

        if self._is_empty:
            # Check 'is Ellipsis' to avoid equality comparison with an array:
            # array equality returns an array, not a boolean.
            if args == () or (len(args) == 1 and args[0] is Ellipsis):
                return Empty(self.dtype)
            raise ValueError("Empty datasets cannot be sliced")

        # Sort field names from the rest of the args.
        names = tuple(x for x in args if isinstance(x, str))

        if names:
            # Read a subset of the fields in this structured dtype
            if len(names) == 1:
                names = names[0]  # Read with simpler dtype of this field
            args = tuple(x for x in args if not isinstance(x, str))
            return self.fields(names, _prior_dtype=new_dtype)[args]

        if new_dtype is None:
            new_dtype = self.dtype
        mtype = h5t.py_create(new_dtype)

        # === Special-case region references ====

        if len(args) == 1 and isinstance(args[0], h5r.RegionReference):

            obj = h5r.dereference(args[0], self.id)
            if obj != self.id:
                raise ValueError("Region reference must point to this dataset")

            sid = h5r.get_region(args[0], self.id)
            mshape = sel.guess_shape(sid)
            if mshape is None:
                # 0D with no data (NULL or deselected SCALAR)
                return Empty(new_dtype)
            out = numpy.empty(mshape, dtype=new_dtype)
            if out.size == 0:
                return out

            sid_out = h5s.create_simple(mshape)
            sid_out.select_all()
            self.id.read(sid_out, sid, out, mtype)
            return out

        # === Check for zero-sized datasets =====

        if self.size == 0:
            # Check 'is Ellipsis' to avoid equality comparison with an array:
            # array equality returns an array, not a boolean.
            if args == () or (len(args) == 1 and args[0] is Ellipsis):
                return numpy.empty(self.shape, dtype=new_dtype)

        # === Scalar dataspaces =================

        if self.shape == ():
            fspace = self.id.get_space()
            selection = sel2.select_read(fspace, args)
            if selection.mshape is None:
                arr = numpy.ndarray((), dtype=new_dtype)
            else:
                arr = numpy.ndarray(selection.mshape, dtype=new_dtype)
            for mspace, fspace in selection:
                self.id.read(mspace, fspace, arr, mtype)
            if selection.mshape is None:
                return arr[()]
            return arr

        # === Everything else ===================

        # Perform the dataspace selection.
        selection = sel.select(self.shape, args, dataset=self)

        if selection.nselect == 0:
            return numpy.ndarray(selection.array_shape, dtype=new_dtype)

        arr = numpy.ndarray(selection.array_shape, new_dtype, order='C')

        # Perform the actual read
        mspace = h5s.create_simple(selection.mshape)
        fspace = selection.id
        self.id.read(mspace, fspace, arr, mtype, dxpl=self._dxpl)

        # Patch up the output for NumPy
        if arr.shape == ():
            return arr[()]   # 0 dim array -> numpy scalar
        return arr

    @with_phil
    def __setitem__(self, args, val):
        """ Write to the HDF5 dataset from a Numpy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the slicing
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        # Generally we try to avoid converting the arrays on the Python
        # side.  However, for compound literals this is unavoidable.
        vlen = h5t.check_vlen_dtype(self.dtype)
        if vlen is not None and vlen not in (bytes, str):
            try:
                val = numpy.asarray(val, dtype=vlen)
            except ValueError:
                try:
                    val = numpy.array([numpy.array(x, dtype=vlen)
                                       for x in val], dtype=self.dtype)
                except ValueError:
                    pass
            if vlen == val.dtype:
                if val.ndim > 1:
                    tmp = numpy.empty(shape=val.shape[:-1], dtype=object)
                    tmp.ravel()[:] = [i for i in val.reshape(
                        (numpy.product(val.shape[:-1], dtype=numpy.ulonglong), val.shape[-1]))]
                else:
                    tmp = numpy.array([None], dtype=object)
                    tmp[0] = val
                val = tmp
        elif self.dtype.kind == "O" or \
          (self.dtype.kind == 'V' and \
          (not isinstance(val, numpy.ndarray) or val.dtype.kind != 'V') and \
          (self.dtype.subdtype is None)):
            if len(names) == 1 and self.dtype.fields is not None:
                # Single field selected for write, from a non-array source
                if not names[0] in self.dtype.fields:
                    raise ValueError("No such field for indexing: %s" % names[0])
                dtype = self.dtype.fields[names[0]][0]
                cast_compound = True
            else:
                dtype = self.dtype
                cast_compound = False

            val = numpy.asarray(val, dtype=dtype.base, order='C')
            if cast_compound:
                val = val.view(numpy.dtype([(names[0], dtype)]))
                val = val.reshape(val.shape[:len(val.shape) - len(dtype.shape)])
        elif (self.dtype.kind == 'S'
              and (h5t.check_string_dtype(self.dtype).encoding == 'utf-8')
              and (find_item_type(val) is str)
        ):
            # Writing str objects to a fixed-length UTF-8 string dataset.
            # Numpy's normal conversion only handles ASCII characters, but
            # when the destination is UTF-8, we want to allow any unicode.
            # This *doesn't* handle numpy fixed-length unicode data ('U' dtype),
            # as HDF5 has no equivalent, and converting fixed length UTF-32
            # to variable length UTF-8 would obscure what's going on.
            str_array = numpy.asarray(val, order='C', dtype=object)
            val = numpy.array([
                s.encode('utf-8') for s in str_array.flat
            ], dtype=self.dtype).reshape(str_array.shape)
        else:
            # If the input data is already an array, let HDF5 do the conversion.
            # If it's a list or similar, don't make numpy guess a dtype for it.
            dt = None if isinstance(val, numpy.ndarray) else self.dtype.base
            val = numpy.asarray(val, order='C', dtype=dt)

        # Check for array dtype compatibility and convert
        if self.dtype.subdtype is not None:
            shp = self.dtype.subdtype[1]
            valshp = val.shape[-len(shp):]
            if valshp != shp:  # Last dimension has to match
                raise TypeError("When writing to array types, last N dimensions have to match (got %s, but should be %s)" % (valshp, shp,))
            mtype = h5t.py_create(numpy.dtype((val.dtype, shp)))
            mshape = val.shape[0:len(val.shape)-len(shp)]

        # Make a compound memory type if field-name slicing is required
        elif len(names) != 0:

            mshape = val.shape

            # Catch common errors
            if self.dtype.fields is None:
                raise TypeError("Illegal slicing argument (not a compound dataset)")
            mismatch = [x for x in names if x not in self.dtype.fields]
            if len(mismatch) != 0:
                mismatch = ", ".join('"%s"'%x for x in mismatch)
                raise ValueError("Illegal slicing argument (fields %s not in dataset type)" % mismatch)

            # Write non-compound source into a single dataset field
            if len(names) == 1 and val.dtype.fields is None:
                subtype = h5t.py_create(val.dtype)
                mtype = h5t.create(h5t.COMPOUND, subtype.get_size())
                mtype.insert(self._e(names[0]), 0, subtype)

            # Make a new source type keeping only the requested fields
            else:
                fieldnames = [x for x in val.dtype.names if x in names] # Keep source order
                mtype = h5t.create(h5t.COMPOUND, val.dtype.itemsize)
                for fieldname in fieldnames:
                    subtype = h5t.py_create(val.dtype.fields[fieldname][0])
                    offset = val.dtype.fields[fieldname][1]
                    mtype.insert(self._e(fieldname), offset, subtype)

        # Use mtype derived from array (let DatasetID.write figure it out)
        else:
            mshape = val.shape
            mtype = None

        # Perform the dataspace selection
        selection = sel.select(self.shape, args, dataset=self)

        if selection.nselect == 0:
            return

        # Broadcast scalars if necessary.
        # In order to avoid slow broadcasting filling the destination by
        # the scalar value, we create an intermediate array of the same
        # size as the destination buffer provided that size is reasonable.
        # We assume as reasonable a size smaller or equal as the used dataset
        # chunk size if any.
        # In case of dealing with a non-chunked destination dataset or with
        # a selection whose size is larger than the dataset chunk size we fall
        # back to using an intermediate array of size equal to the last dimension
        # of the destination buffer.
        # The reasoning behind is that it makes sense to assume the creator of
        # the dataset used an appropriate chunk size according the available
        # memory. In any case, if we cannot afford to create an intermediate
        # array of the same size as the dataset chunk size, the user program has
        # little hope to go much further. Solves h5py issue #1067
        if mshape == () and selection.array_shape != ():
            if self.dtype.subdtype is not None:
                raise TypeError("Scalar broadcasting is not supported for array dtypes")
            if self.chunks and (numpy.prod(self.chunks, dtype=numpy.float64) >=
                                numpy.prod(selection.array_shape, dtype=numpy.float64)):
                val2 = numpy.empty(selection.array_shape, dtype=val.dtype)
            else:
                val2 = numpy.empty(selection.array_shape[-1], dtype=val.dtype)
            val2[...] = val
            val = val2
            mshape = val.shape

        # Perform the write, with broadcasting
        mspace = h5s.create_simple(selection.expand_shape(mshape))
        for fspace in selection.broadcast(mshape):
            self.id.write(mspace, fspace, val, mtype, dxpl=self._dxpl)

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """ Read data directly from HDF5 into an existing NumPy array.

        The destination array must be C-contiguous and writable.
        Selections must be the output of numpy.s_[<args>].

        Broadcasting is supported for simple indexing.
        """
        with phil:
            if self._is_empty:
                raise TypeError("Empty datasets have no numpy representation")
            if source_sel is None:
                source_sel = sel.SimpleSelection(self.shape)
            else:
                source_sel = sel.select(self.shape, source_sel, self)  # for numpy.s_
            fspace = source_sel.id

            if dest_sel is None:
                dest_sel = sel.SimpleSelection(dest.shape)
            else:
                dest_sel = sel.select(dest.shape, dest_sel)

            for mspace in dest_sel.broadcast(source_sel.array_shape):
                self.id.read(mspace, fspace, dest, dxpl=self._dxpl)

    def write_direct(self, source, source_sel=None, dest_sel=None):
        """ Write data directly to HDF5 from a NumPy array.

        The source array must be C-contiguous.  Selections must be
        the output of numpy.s_[<args>].

        Broadcasting is supported for simple indexing.
        """
        with phil:
            if self._is_empty:
                raise TypeError("Empty datasets cannot be written to")
            if source_sel is None:
                source_sel = sel.SimpleSelection(source.shape)
            else:
                source_sel = sel.select(source.shape, source_sel)  # for numpy.s_
            mspace = source_sel.id

            if dest_sel is None:
                dest_sel = sel.SimpleSelection(self.shape)
            else:
                dest_sel = sel.select(self.shape, dest_sel, self)

            for fspace in dest_sel.broadcast(source_sel.array_shape):
                self.id.write(mspace, fspace, source, dxpl=self._dxpl)

    @with_phil
    def __array__(self, dtype=None):
        """ Create a Numpy array containing the whole dataset.  DON'T THINK
        THIS MEANS DATASETS ARE INTERCHANGEABLE WITH ARRAYS.  For one thing,
        you have to read the whole dataset every time this method is called.
        """
        arr = numpy.empty(self.shape, dtype=self.dtype if dtype is None else dtype)

        # Special case for (0,)*-shape datasets
        if numpy.product(self.shape, dtype=numpy.ulonglong) == 0:
            return arr

        self.read_direct(arr)
        return arr

    @with_phil
    def __repr__(self):
        if not self:
            r = '<Closed HDF5 dataset>'
        else:
            if self.name is None:
                namestr = '("anonymous")'
            else:
                name = pp.basename(pp.normpath(self.name))
                namestr = '"%s"' % (name if name != '' else '/')
            r = '<HDF5 dataset %s: shape %s, type "%s">' % (
                namestr, self.shape, self.dtype.str
            )
        return r

    if hasattr(h5d.DatasetID, "refresh"):
        @with_phil
        def refresh(self):
            """ Refresh the dataset metadata by reloading from the file.

            This is part of the SWMR features and only exist when the HDF5
            library version >=1.9.178
            """
            self._id.refresh()
            self._cache_props.clear()

    if hasattr(h5d.DatasetID, "flush"):
        @with_phil
        def flush(self):
            """ Flush the dataset data and metadata to the file.
            If the dataset is chunked, raw data chunks are written to the file.

            This is part of the SWMR features and only exist when the HDF5
            library version >=1.9.178
            """
            self._id.flush()

    if vds_support:
        @property
        @with_phil
        def is_virtual(self):
            """Check if this is a virtual dataset"""
            return self._dcpl.get_layout() == h5d.VIRTUAL

        @with_phil
        def virtual_sources(self):
            """Get a list of the data mappings for a virtual dataset"""
            if not self.is_virtual:
                raise RuntimeError("Not a virtual dataset")
            dcpl = self._dcpl
            return [
                VDSmap(dcpl.get_virtual_vspace(j),
                       dcpl.get_virtual_filename(j),
                       dcpl.get_virtual_dsetname(j),
                       dcpl.get_virtual_srcspace(j))
                for j in range(dcpl.get_virtual_count())]

    @with_phil
    def make_scale(self, name=''):
        """Make this dataset an HDF5 dimension scale.

        You can then attach it to dimensions of other datasets like this::

            other_ds.dims[0].attach_scale(ds)

        You can optionally pass a name to associate with this scale.
        """
        h5ds.set_scale(self._id, self._e(name))
