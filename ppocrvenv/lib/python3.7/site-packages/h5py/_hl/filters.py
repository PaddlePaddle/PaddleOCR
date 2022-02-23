# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements support for HDF5 compression filters via the high-level
    interface.  The following types of filter are available:

    "gzip"
        Standard DEFLATE-based compression, at integer levels from 0 to 9.
        Built-in to all public versions of HDF5.  Use this if you want a
        decent-to-good ratio, good portability, and don't mind waiting.

    "lzf"
        Custom compression filter for h5py.  This filter is much, much faster
        than gzip (roughly 10x in compression vs. gzip level 4, and 3x faster
        in decompressing), but at the cost of a worse compression ratio.  Use
        this if you want cheap compression and portability is not a concern.

    "szip"
        Access to the HDF5 SZIP encoder.  SZIP is a non-mainstream compression
        format used in space science on integer and float datasets.  SZIP is
        subject to license requirements, which means the encoder is not
        guaranteed to be always available.  However, it is also much faster
        than gzip.

    The following constants in this module are also useful:

    decode
        Tuple of available filter names for decoding

    encode
        Tuple of available filter names for encoding
"""
from collections.abc import Mapping
import operator

import numpy as np
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f


_COMP_FILTERS = {'gzip': h5z.FILTER_DEFLATE,
                'szip': h5z.FILTER_SZIP,
                'lzf': h5z.FILTER_LZF,
                'shuffle': h5z.FILTER_SHUFFLE,
                'fletcher32': h5z.FILTER_FLETCHER32,
                'scaleoffset': h5z.FILTER_SCALEOFFSET }

DEFAULT_GZIP = 4
DEFAULT_SZIP = ('nn', 8)

def _gen_filter_tuples():
    """ Bootstrap function to figure out what filters are available. """
    dec = []
    enc = []
    for name, code in _COMP_FILTERS.items():
        if h5z.filter_avail(code):
            info = h5z.get_filter_info(code)
            if info & h5z.FILTER_CONFIG_ENCODE_ENABLED:
                enc.append(name)
            if info & h5z.FILTER_CONFIG_DECODE_ENABLED:
                dec.append(name)

    return tuple(dec), tuple(enc)

decode, encode = _gen_filter_tuples()

def _external_entry(entry):
    """ Check for and return a well-formed entry tuple for
    a call to h5p.set_external. """
    # We require only an iterable entry but also want to guard against
    # raising a confusing exception from unpacking below a str or bytes that
    # was mistakenly passed as an entry.  We go further than that and accept
    # only a tuple, which allows simpler documentation and exception
    # messages.
    if not isinstance(entry, tuple):
        raise TypeError(
            "Each external entry must be a tuple of (name, offset, size)")
    name, offset, size = entry  # raise ValueError without three elements
    name = filename_encode(name)
    offset = operator.index(offset)
    size = operator.index(size)
    return (name, offset, size)

def _normalize_external(external):
    """ Normalize external into a well-formed list of tuples and return. """
    if external is None:
        return []
    try:
        # Accept a solitary name---a str, bytes, or os.PathLike acceptable to
        # filename_encode.
        return [_external_entry((external, 0, h5f.UNLIMITED))]
    except TypeError:
        pass
    # Check and rebuild each entry to be well-formed.
    return [_external_entry(entry) for entry in external]

class FilterRefBase(Mapping):
    """Base class for referring to an HDF5 and describing its options

    Your subclass must define filter_id, and may define a filter_options tuple.
    """
    filter_id = None
    filter_options = ()

    # Mapping interface supports using instances as **kwargs for compatibility
    # with older versions of h5py
    @property
    def _kwargs(self):
        return {
            'compression': self.filter_id,
            'compression_opts': self.filter_options
        }

    def __hash__(self):
        return hash((self.filter_id, self.filter_options))

    def __eq__(self, other):
        return (
            isinstance(other, FilterRefBase)
            and self.filter_id == other.filter_id
            and self.filter_options == other.filter_options
        )

    def __len__(self):
        return len(self._kwargs)

    def __iter__(self):
        return iter(self._kwargs)

    def __getitem__(self, item):
        return self._kwargs[item]

class Gzip(FilterRefBase):
    filter_id = h5z.FILTER_DEFLATE

    def __init__(self, level=DEFAULT_GZIP):
        self.filter_options = (level,)

def fill_dcpl(plist, shape, dtype, chunks, compression, compression_opts,
              shuffle, fletcher32, maxshape, scaleoffset, external,
              allow_unknown_filter=False):
    """ Generate a dataset creation property list.

    Undocumented and subject to change without warning.
    """

    if shape is None or shape == ():
        shapetype = 'Empty' if shape is None else 'Scalar'
        if any((chunks, compression, compression_opts, shuffle, fletcher32,
                scaleoffset is not None)):
            raise TypeError(
                f"{shapetype} datasets don't support chunk/filter options"
            )
        if maxshape and maxshape != ():
            raise TypeError(f"{shapetype} datasets cannot be extended")
        return h5p.create(h5p.DATASET_CREATE)

    def rq_tuple(tpl, name):
        """ Check if chunks/maxshape match dataset rank """
        if tpl in (None, True):
            return
        try:
            tpl = tuple(tpl)
        except TypeError:
            raise TypeError('"%s" argument must be None or a sequence object' % name)
        if len(tpl) != len(shape):
            raise ValueError('"%s" must have same rank as dataset shape' % name)

    rq_tuple(chunks, 'chunks')
    rq_tuple(maxshape, 'maxshape')

    if compression is not None:
        if isinstance(compression, FilterRefBase):
            compression_opts = compression.filter_options
            compression = compression.filter_id

        if compression not in encode and not isinstance(compression, int):
            raise ValueError('Compression filter "%s" is unavailable' % compression)

        if compression == 'gzip':
            if compression_opts is None:
                gzip_level = DEFAULT_GZIP
            elif compression_opts in range(10):
                gzip_level = compression_opts
            else:
                raise ValueError("GZIP setting must be an integer from 0-9, not %r" % compression_opts)

        elif compression == 'lzf':
            if compression_opts is not None:
                raise ValueError("LZF compression filter accepts no options")

        elif compression == 'szip':
            if compression_opts is None:
                compression_opts = DEFAULT_SZIP

            err = "SZIP options must be a 2-tuple ('ec'|'nn', even integer 0-32)"
            try:
                szmethod, szpix = compression_opts
            except TypeError:
                raise TypeError(err)
            if szmethod not in ('ec', 'nn'):
                raise ValueError(err)
            if not (0<szpix<=32 and szpix%2 == 0):
                raise ValueError(err)

    elif compression_opts is not None:
        # Can't specify just compression_opts by itself.
        raise TypeError("Compression method must be specified")

    if scaleoffset is not None:
        # scaleoffset must be an integer when it is not None or False,
        # except for integral data, for which scaleoffset == True is
        # permissible (will use SO_INT_MINBITS_DEFAULT)

        if scaleoffset < 0:
            raise ValueError('scale factor must be >= 0')

        if dtype.kind == 'f':
            if scaleoffset is True:
                raise ValueError('integer scaleoffset must be provided for '
                                 'floating point types')
        elif dtype.kind in ('u', 'i'):
            if scaleoffset is True:
                scaleoffset = h5z.SO_INT_MINBITS_DEFAULT
        else:
            raise TypeError('scale/offset filter only supported for integer '
                            'and floating-point types')

        # Scale/offset following fletcher32 in the filter chain will (almost?)
        # always triggers a read error, as most scale/offset settings are
        # lossy. Since fletcher32 must come first (see comment below) we
        # simply prohibit the combination of fletcher32 and scale/offset.
        if fletcher32:
            raise ValueError('fletcher32 cannot be used with potentially lossy'
                             ' scale/offset filter')

    external = _normalize_external(external)
    # End argument validation

    if (chunks is True) or \
    (chunks is None and any((shuffle, fletcher32, compression, maxshape,
                             scaleoffset is not None))):
        chunks = guess_chunk(shape, maxshape, dtype.itemsize)

    if maxshape is True:
        maxshape = (None,)*len(shape)

    if chunks is not None:
        plist.set_chunk(chunks)
        plist.set_fill_time(h5d.FILL_TIME_ALLOC)  # prevent resize glitch

    # scale-offset must come before shuffle and compression
    if scaleoffset is not None:
        if dtype.kind in ('u', 'i'):
            plist.set_scaleoffset(h5z.SO_INT, scaleoffset)
        else: # dtype.kind == 'f'
            plist.set_scaleoffset(h5z.SO_FLOAT_DSCALE, scaleoffset)

    for item in external:
        plist.set_external(*item)

    if shuffle:
        plist.set_shuffle()

    if compression == 'gzip':
        plist.set_deflate(gzip_level)
    elif compression == 'lzf':
        plist.set_filter(h5z.FILTER_LZF, h5z.FLAG_OPTIONAL)
    elif compression == 'szip':
        opts = {'ec': h5z.SZIP_EC_OPTION_MASK, 'nn': h5z.SZIP_NN_OPTION_MASK}
        plist.set_szip(opts[szmethod], szpix)
    elif isinstance(compression, int):
        if not allow_unknown_filter and not h5z.filter_avail(compression):
            raise ValueError("Unknown compression filter number: %s" % compression)

        plist.set_filter(compression, h5z.FLAG_OPTIONAL, compression_opts)

    # `fletcher32` must come after `compression`, otherwise, if `compression`
    # is "szip" and the data is 64bit, the fletcher32 checksum will be wrong
    # (see GitHub issue #953).
    if fletcher32:
        plist.set_fletcher32()

    return plist

def get_filters(plist):
    """ Extract a dictionary of active filters from a DCPL, along with
    their settings.

    Undocumented and subject to change without warning.
    """

    filters = {h5z.FILTER_DEFLATE: 'gzip', h5z.FILTER_SZIP: 'szip',
               h5z.FILTER_SHUFFLE: 'shuffle', h5z.FILTER_FLETCHER32: 'fletcher32',
               h5z.FILTER_LZF: 'lzf', h5z.FILTER_SCALEOFFSET: 'scaleoffset'}

    pipeline = {}

    nfilters = plist.get_nfilters()

    for i in range(nfilters):

        code, _, vals, _ = plist.get_filter(i)

        if code == h5z.FILTER_DEFLATE:
            vals = vals[0] # gzip level

        elif code == h5z.FILTER_SZIP:
            mask, pixels = vals[0:2]
            if mask & h5z.SZIP_EC_OPTION_MASK:
                mask = 'ec'
            elif mask & h5z.SZIP_NN_OPTION_MASK:
                mask = 'nn'
            else:
                raise TypeError("Unknown SZIP configuration")
            vals = (mask, pixels)
        elif code == h5z.FILTER_LZF:
            vals = None
        else:
            if len(vals) == 0:
                vals = None

        pipeline[filters.get(code, str(code))] = vals

    return pipeline

CHUNK_BASE = 16*1024    # Multiplier by which chunks are adjusted
CHUNK_MIN = 8*1024      # Soft lower limit (8k)
CHUNK_MAX = 1024*1024   # Hard upper limit (1M)

def guess_chunk(shape, maxshape, typesize):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.

    Undocumented and subject to change without warning.
    """
    # pylint: disable=unused-argument

    # For unlimited dimensions we have to guess 1024
    shape = tuple((x if x!=0 else 1024) for i, x in enumerate(shape))

    ndims = len(shape)
    if ndims == 0:
        raise ValueError("Chunks not allowed for scalar datasets.")

    chunks = np.array(shape, dtype='=f8')
    if not np.all(np.isfinite(chunks)):
        raise ValueError("Illegal value in chunk tuple")

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.product(chunks)*typesize
    target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))

    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        #  2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.product(chunks)*typesize

        if (chunk_bytes < target_size or \
         abs(chunk_bytes-target_size)/target_size < 0.5) and \
         chunk_bytes < CHUNK_MAX:
            break

        if np.product(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        chunks[idx%ndims] = np.ceil(chunks[idx%ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)
