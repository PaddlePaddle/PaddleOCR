"""
NetCDF reader/writer module.

This module is used to read and create NetCDF files. NetCDF files are
accessed through the `netcdf_file` object. Data written to and from NetCDF
files are contained in `netcdf_variable` objects. Attributes are given
as member variables of the `netcdf_file` and `netcdf_variable` objects.

This module implements the Scientific.IO.NetCDF API to read and create
NetCDF files. The same API is also used in the PyNIO and pynetcdf
modules, allowing these modules to be used interchangeably when working
with NetCDF files.

Only NetCDF3 is supported here; for NetCDF4 see
`netCDF4-python <http://unidata.github.io/netcdf4-python/>`__,
which has a similar API.

"""

# TODO:
# * properly implement ``_FillValue``.
# * fix character variables.
# * implement PAGESIZE for Python 2.6?

# The Scientific.IO.NetCDF API allows attributes to be added directly to
# instances of ``netcdf_file`` and ``netcdf_variable``. To differentiate
# between user-set attributes and instance attributes, user-set attributes
# are automatically stored in the ``_attributes`` attribute by overloading
#``__setattr__``. This is the reason why the code sometimes uses
#``obj.__dict__['key'] = value``, instead of simply ``obj.key = value``;
# otherwise the key would be inserted into userspace attributes.


__all__ = ['netcdf_file', 'netcdf_variable']


import warnings
import weakref
from operator import mul
from platform import python_implementation

import mmap as mm

import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce


IS_PYPY = python_implementation() == 'PyPy'

ABSENT = b'\x00\x00\x00\x00\x00\x00\x00\x00'
ZERO = b'\x00\x00\x00\x00'
NC_BYTE = b'\x00\x00\x00\x01'
NC_CHAR = b'\x00\x00\x00\x02'
NC_SHORT = b'\x00\x00\x00\x03'
NC_INT = b'\x00\x00\x00\x04'
NC_FLOAT = b'\x00\x00\x00\x05'
NC_DOUBLE = b'\x00\x00\x00\x06'
NC_DIMENSION = b'\x00\x00\x00\n'
NC_VARIABLE = b'\x00\x00\x00\x0b'
NC_ATTRIBUTE = b'\x00\x00\x00\x0c'
FILL_BYTE = b'\x81'
FILL_CHAR = b'\x00'
FILL_SHORT = b'\x80\x01'
FILL_INT = b'\x80\x00\x00\x01'
FILL_FLOAT = b'\x7C\xF0\x00\x00'
FILL_DOUBLE = b'\x47\x9E\x00\x00\x00\x00\x00\x00'

TYPEMAP = {NC_BYTE: ('b', 1),
           NC_CHAR: ('c', 1),
           NC_SHORT: ('h', 2),
           NC_INT: ('i', 4),
           NC_FLOAT: ('f', 4),
           NC_DOUBLE: ('d', 8)}

FILLMAP = {NC_BYTE: FILL_BYTE,
           NC_CHAR: FILL_CHAR,
           NC_SHORT: FILL_SHORT,
           NC_INT: FILL_INT,
           NC_FLOAT: FILL_FLOAT,
           NC_DOUBLE: FILL_DOUBLE}

REVERSE = {('b', 1): NC_BYTE,
           ('B', 1): NC_CHAR,
           ('c', 1): NC_CHAR,
           ('h', 2): NC_SHORT,
           ('i', 4): NC_INT,
           ('f', 4): NC_FLOAT,
           ('d', 8): NC_DOUBLE,

           # these come from asarray(1).dtype.char and asarray('foo').dtype.char,
           # used when getting the types from generic attributes.
           ('l', 4): NC_INT,
           ('S', 1): NC_CHAR}


class netcdf_file:
    """
    A file object for NetCDF data.

    A `netcdf_file` object has two standard attributes: `dimensions` and
    `variables`. The values of both are dictionaries, mapping dimension
    names to their associated lengths and variable names to variables,
    respectively. Application programs should never modify these
    dictionaries.

    All other attributes correspond to global attributes defined in the
    NetCDF file. Global file attributes are created by assigning to an
    attribute of the `netcdf_file` object.

    Parameters
    ----------
    filename : string or file-like
        string -> filename
    mode : {'r', 'w', 'a'}, optional
        read-write-append mode, default is 'r'
    mmap : None or bool, optional
        Whether to mmap `filename` when reading.  Default is True
        when `filename` is a file name, False when `filename` is a
        file-like object. Note that when mmap is in use, data arrays
        returned refer directly to the mmapped data on disk, and the
        file cannot be closed as long as references to it exist.
    version : {1, 2}, optional
        version of netcdf to read / write, where 1 means *Classic
        format* and 2 means *64-bit offset format*.  Default is 1.  See
        `here <https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html#select_format>`__
        for more info.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.

    Notes
    -----
    The major advantage of this module over other modules is that it doesn't
    require the code to be linked to the NetCDF libraries. This module is
    derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.

    NetCDF files are a self-describing binary data format. The file contains
    metadata that describes the dimensions and variables in the file. More
    details about NetCDF files can be found `here
    <https://www.unidata.ucar.edu/software/netcdf/guide_toc.html>`__. There
    are three main sections to a NetCDF data structure:

    1. Dimensions
    2. Variables
    3. Attributes

    The dimensions section records the name and length of each dimension used
    by the variables. The variables would then indicate which dimensions it
    uses and any attributes such as data units, along with containing the data
    values for the variable. It is good practice to include a
    variable that is the same name as a dimension to provide the values for
    that axes. Lastly, the attributes section would contain additional
    information such as the name of the file creator or the instrument used to
    collect the data.

    When writing data to a NetCDF file, there is often the need to indicate the
    'record dimension'. A record dimension is the unbounded dimension for a
    variable. For example, a temperature variable may have dimensions of
    latitude, longitude and time. If one wants to add more temperature data to
    the NetCDF file as time progresses, then the temperature variable should
    have the time dimension flagged as the record dimension.

    In addition, the NetCDF file header contains the position of the data in
    the file, so access can be done in an efficient manner without loading
    unnecessary data into memory. It uses the ``mmap`` module to create
    Numpy arrays mapped to the data on disk, for the same purpose.

    Note that when `netcdf_file` is used to open a file with mmap=True
    (default for read-only), arrays returned by it refer to data
    directly on the disk. The file should not be closed, and cannot be cleanly
    closed when asked, if such arrays are alive. You may want to copy data arrays
    obtained from mmapped Netcdf file if they are to be processed after the file
    is closed, see the example below.

    Examples
    --------
    To create a NetCDF file:

    >>> from scipy.io import netcdf
    >>> f = netcdf.netcdf_file('simple.nc', 'w')
    >>> f.history = 'Created for a test'
    >>> f.createDimension('time', 10)
    >>> time = f.createVariable('time', 'i', ('time',))
    >>> time[:] = np.arange(10)
    >>> time.units = 'days since 2008-01-01'
    >>> f.close()

    Note the assignment of ``arange(10)`` to ``time[:]``.  Exposing the slice
    of the time variable allows for the data to be set in the object, rather
    than letting ``arange(10)`` overwrite the ``time`` variable.

    To read the NetCDF file we just created:

    >>> from scipy.io import netcdf
    >>> f = netcdf.netcdf_file('simple.nc', 'r')
    >>> print(f.history)
    b'Created for a test'
    >>> time = f.variables['time']
    >>> print(time.units)
    b'days since 2008-01-01'
    >>> print(time.shape)
    (10,)
    >>> print(time[-1])
    9

    NetCDF files, when opened read-only, return arrays that refer
    directly to memory-mapped data on disk:

    >>> data = time[:]
    >>> data.base.base
    <mmap.mmap object at 0x7fe753763180>

    If the data is to be processed after the file is closed, it needs
    to be copied to main memory:

    >>> data = time[:].copy()
    >>> f.close()
    >>> data.mean()
    4.5

    A NetCDF file can also be used as context manager:

    >>> from scipy.io import netcdf
    >>> with netcdf.netcdf_file('simple.nc', 'r') as f:
    ...     print(f.history)
    b'Created for a test'

    """
    def __init__(self, filename, mode='r', mmap=None, version=1,
                 maskandscale=False):
        """Initialize netcdf_file from fileobj (str or file-like)."""
        if mode not in 'rwa':
            raise ValueError("Mode must be either 'r', 'w' or 'a'.")

        if hasattr(filename, 'seek'):  # file-like
            self.fp = filename
            self.filename = 'None'
            if mmap is None:
                mmap = False
            elif mmap and not hasattr(filename, 'fileno'):
                raise ValueError('Cannot use file object for mmap')
        else:  # maybe it's a string
            self.filename = filename
            omode = 'r+' if mode == 'a' else mode
            self.fp = open(self.filename, '%sb' % omode)
            if mmap is None:
                # Mmapped files on PyPy cannot be usually closed
                # before the GC runs, so it's better to use mmap=False
                # as the default.
                mmap = (not IS_PYPY)

        if mode != 'r':
            # Cannot read write-only files
            mmap = False

        self.use_mmap = mmap
        self.mode = mode
        self.version_byte = version
        self.maskandscale = maskandscale

        self.dimensions = {}
        self.variables = {}

        self._dims = []
        self._recs = 0
        self._recsize = 0

        self._mm = None
        self._mm_buf = None
        if self.use_mmap:
            self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
            self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)

        self._attributes = {}

        if mode in 'ra':
            self._read()

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def close(self):
        """Closes the NetCDF file."""
        if hasattr(self, 'fp') and not self.fp.closed:
            try:
                self.flush()
            finally:
                self.variables = {}
                if self._mm_buf is not None:
                    ref = weakref.ref(self._mm_buf)
                    self._mm_buf = None
                    if ref() is None:
                        # self._mm_buf is gc'd, and we can close the mmap
                        self._mm.close()
                    else:
                        # we cannot close self._mm, since self._mm_buf is
                        # alive and there may still be arrays referring to it
                        warnings.warn((
                            "Cannot close a netcdf_file opened with mmap=True, when "
                            "netcdf_variables or arrays referring to its data still exist. "
                            "All data arrays obtained from such files refer directly to "
                            "data on disk, and must be copied before the file can be cleanly "
                            "closed. (See netcdf_file docstring for more information on mmap.)"
                        ), category=RuntimeWarning)
                self._mm = None
                self.fp.close()
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def createDimension(self, name, length):
        """
        Adds a dimension to the Dimension section of the NetCDF data structure.

        Note that this function merely adds a new dimension that the variables can
        reference. The values for the dimension, if desired, should be added as
        a variable using `createVariable`, referring to this dimension.

        Parameters
        ----------
        name : str
            Name of the dimension (Eg, 'lat' or 'time').
        length : int
            Length of the dimension.

        See Also
        --------
        createVariable

        """
        if length is None and self._dims:
            raise ValueError("Only first dimension may be unlimited!")

        self.dimensions[name] = length
        self._dims.append(name)

    def createVariable(self, name, type, dimensions):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        shape = tuple([self.dimensions[dim] for dim in dimensions])
        shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for NumPy

        type = dtype(type)
        typecode, size = type.char, type.itemsize
        if (typecode, size) not in REVERSE:
            raise ValueError("NetCDF 3 does not support type %s" % type)

        data = empty(shape_, dtype=type.newbyteorder("B"))  # convert to big endian always for NetCDF 3
        self.variables[name] = netcdf_variable(
                data, typecode, size, shape, dimensions,
                maskandscale=self.maskandscale)
        return self.variables[name]

    def flush(self):
        """
        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

        See Also
        --------
        sync : Identical function

        """
        if hasattr(self, 'mode') and self.mode in 'wa':
            self._write()
    sync = flush

    def _write(self):
        self.fp.seek(0)
        self.fp.write(b'CDF')
        self.fp.write(array(self.version_byte, '>b').tobytes())

        # Write headers and data.
        self._write_numrecs()
        self._write_dim_array()
        self._write_gatt_array()
        self._write_var_array()

    def _write_numrecs(self):
        # Get highest record count from all record variables.
        for var in self.variables.values():
            if var.isrec and len(var.data) > self._recs:
                self.__dict__['_recs'] = len(var.data)
        self._pack_int(self._recs)

    def _write_dim_array(self):
        if self.dimensions:
            self.fp.write(NC_DIMENSION)
            self._pack_int(len(self.dimensions))
            for name in self._dims:
                self._pack_string(name)
                length = self.dimensions[name]
                self._pack_int(length or 0)  # replace None with 0 for record dimension
        else:
            self.fp.write(ABSENT)

    def _write_gatt_array(self):
        self._write_att_array(self._attributes)

    def _write_att_array(self, attributes):
        if attributes:
            self.fp.write(NC_ATTRIBUTE)
            self._pack_int(len(attributes))
            for name, values in attributes.items():
                self._pack_string(name)
                self._write_att_values(values)
        else:
            self.fp.write(ABSENT)

    def _write_var_array(self):
        if self.variables:
            self.fp.write(NC_VARIABLE)
            self._pack_int(len(self.variables))

            # Sort variable names non-recs first, then recs.
            def sortkey(n):
                v = self.variables[n]
                if v.isrec:
                    return (-1,)
                return v._shape
            variables = sorted(self.variables, key=sortkey, reverse=True)

            # Set the metadata for all variables.
            for name in variables:
                self._write_var_metadata(name)
            # Now that we have the metadata, we know the vsize of
            # each record variable, so we can calculate recsize.
            self.__dict__['_recsize'] = sum([
                    var._vsize for var in self.variables.values()
                    if var.isrec])
            # Set the data for all variables.
            for name in variables:
                self._write_var_data(name)
        else:
            self.fp.write(ABSENT)

    def _write_var_metadata(self, name):
        var = self.variables[name]

        self._pack_string(name)
        self._pack_int(len(var.dimensions))
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            self._pack_int(dimid)

        self._write_att_array(var._attributes)

        nc_type = REVERSE[var.typecode(), var.itemsize()]
        self.fp.write(nc_type)

        if not var.isrec:
            vsize = var.data.size * var.data.itemsize
            vsize += -vsize % 4
        else:  # record variable
            try:
                vsize = var.data[0].size * var.data.itemsize
            except IndexError:
                vsize = 0
            rec_vars = len([v for v in self.variables.values()
                            if v.isrec])
            if rec_vars > 1:
                vsize += -vsize % 4
        self.variables[name].__dict__['_vsize'] = vsize
        self._pack_int(vsize)

        # Pack a bogus begin, and set the real value later.
        self.variables[name].__dict__['_begin'] = self.fp.tell()
        self._pack_begin(0)

    def _write_var_data(self, name):
        var = self.variables[name]

        # Set begin in file header.
        the_beguine = self.fp.tell()
        self.fp.seek(var._begin)
        self._pack_begin(the_beguine)
        self.fp.seek(the_beguine)

        # Write data.
        if not var.isrec:
            self.fp.write(var.data.tobytes())
            count = var.data.size * var.data.itemsize
            self._write_var_padding(var, var._vsize - count)
        else:  # record variable
            # Handle rec vars with shape[0] < nrecs.
            if self._recs > len(var.data):
                shape = (self._recs,) + var.data.shape[1:]
                # Resize in-place does not always work since
                # the array might not be single-segment
                try:
                    var.data.resize(shape)
                except ValueError:
                    var.__dict__['data'] = np.resize(var.data, shape).astype(var.data.dtype)

            pos0 = pos = self.fp.tell()
            for rec in var.data:
                # Apparently scalars cannot be converted to big endian. If we
                # try to convert a ``=i4`` scalar to, say, '>i4' the dtype
                # will remain as ``=i4``.
                if not rec.shape and (rec.dtype.byteorder == '<' or
                        (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                    rec = rec.byteswap()
                self.fp.write(rec.tobytes())
                # Padding
                count = rec.size * rec.itemsize
                self._write_var_padding(var, var._vsize - count)
                pos += self._recsize
                self.fp.seek(pos)
            self.fp.seek(pos0 + var._vsize)

    def _write_var_padding(self, var, size):
        encoded_fill_value = var._get_encoded_fill_value()
        num_fills = size // len(encoded_fill_value)
        self.fp.write(encoded_fill_value * num_fills)

    def _write_att_values(self, values):
        if hasattr(values, 'dtype'):
            nc_type = REVERSE[values.dtype.char, values.dtype.itemsize]
        else:
            types = [(int, NC_INT), (float, NC_FLOAT), (str, NC_CHAR)]

            # bytes index into scalars in py3k. Check for "string" types
            if isinstance(values, (str, bytes)):
                sample = values
            else:
                try:
                    sample = values[0]  # subscriptable?
                except TypeError:
                    sample = values     # scalar

            for class_, nc_type in types:
                if isinstance(sample, class_):
                    break

        typecode, size = TYPEMAP[nc_type]
        dtype_ = '>%s' % typecode
        # asarray() dies with bytes and '>c' in py3k. Change to 'S'
        dtype_ = 'S' if dtype_ == '>c' else dtype_

        values = asarray(values, dtype=dtype_)

        self.fp.write(nc_type)

        if values.dtype.char == 'S':
            nelems = values.itemsize
        else:
            nelems = values.size
        self._pack_int(nelems)

        if not values.shape and (values.dtype.byteorder == '<' or
                (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
            values = values.byteswap()
        self.fp.write(values.tobytes())
        count = values.size * values.itemsize
        self.fp.write(b'\x00' * (-count % 4))  # pad

    def _read(self):
        # Check magic bytes and version
        magic = self.fp.read(3)
        if not magic == b'CDF':
            raise TypeError("Error: %s is not a valid NetCDF 3 file" %
                            self.filename)
        self.__dict__['version_byte'] = frombuffer(self.fp.read(1), '>b')[0]

        # Read file headers and set data.
        self._read_numrecs()
        self._read_dim_array()
        self._read_gatt_array()
        self._read_var_array()

    def _read_numrecs(self):
        self.__dict__['_recs'] = self._unpack_int()

    def _read_dim_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_DIMENSION]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        for dim in range(count):
            name = self._unpack_string().decode('latin1')
            length = self._unpack_int() or None  # None for record dimension
            self.dimensions[name] = length
            self._dims.append(name)  # preserve order

    def _read_gatt_array(self):
        for k, v in self._read_att_array().items():
            self.__setattr__(k, v)

    def _read_att_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_ATTRIBUTE]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        attributes = {}
        for attr in range(count):
            name = self._unpack_string().decode('latin1')
            attributes[name] = self._read_att_values()
        return attributes

    def _read_var_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_VARIABLE]:
            raise ValueError("Unexpected header.")

        begin = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            (name, dimensions, shape, attributes,
             typecode, size, dtype_, begin_, vsize) = self._read_var()
            # https://www.unidata.ucar.edu/software/netcdf/guide_toc.html
            # Note that vsize is the product of the dimension lengths
            # (omitting the record dimension) and the number of bytes
            # per value (determined from the type), increased to the
            # next multiple of 4, for each variable. If a record
            # variable, this is the amount of space per record. The
            # netCDF "record size" is calculated as the sum of the
            # vsize's of all the record variables.
            #
            # The vsize field is actually redundant, because its value
            # may be computed from other information in the header. The
            # 32-bit vsize field is not large enough to contain the size
            # of variables that require more than 2^32 - 4 bytes, so
            # 2^32 - 1 is used in the vsize field for such variables.
            if shape and shape[0] is None:  # record variable
                rec_vars.append(name)
                # The netCDF "record size" is calculated as the sum of
                # the vsize's of all the record variables.
                self.__dict__['_recsize'] += vsize
                if begin == 0:
                    begin = begin_
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + dtype_)

                # Handle padding with a virtual variable.
                if typecode in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % var)
                        dtypes['formats'].append('(%d,)>b' % padding)

                # Data will be set later.
                data = None
            else:  # not a record variable
                # Calculate size to avoid problems with vsize (above)
                a_size = reduce(mul, shape, 1) * size
                if self.use_mmap:
                    data = self._mm_buf[begin_:begin_+a_size].view(dtype=dtype_)
                    data.shape = shape
                else:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = frombuffer(self.fp.read(a_size), dtype=dtype_
                                      ).copy()
                    data.shape = shape
                    self.fp.seek(pos)

            # Add variable.
            self.variables[name] = netcdf_variable(
                    data, typecode, size, shape, dimensions, attributes,
                    maskandscale=self.maskandscale)

        if rec_vars:
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            # Build rec array.
            if self.use_mmap:
                rec_array = self._mm_buf[begin:begin+self._recs*self._recsize].view(dtype=dtypes)
                rec_array.shape = (self._recs,)
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                rec_array = frombuffer(self.fp.read(self._recs*self._recsize),
                                       dtype=dtypes).copy()
                rec_array.shape = (self._recs,)
                self.fp.seek(pos)

            for var in rec_vars:
                self.variables[var].__dict__['data'] = rec_array[var]

    def _read_var(self):
        name = self._unpack_string().decode('latin1')
        dimensions = []
        shape = []
        dims = self._unpack_int()

        for i in range(dims):
            dimid = self._unpack_int()
            dimname = self._dims[dimid]
            dimensions.append(dimname)
            dim = self.dimensions[dimname]
            shape.append(dim)
        dimensions = tuple(dimensions)
        shape = tuple(shape)

        attributes = self._read_att_array()
        nc_type = self.fp.read(4)
        vsize = self._unpack_int()
        begin = [self._unpack_int, self._unpack_int64][self.version_byte-1]()

        typecode, size = TYPEMAP[nc_type]
        dtype_ = '>%s' % typecode

        return name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize

    def _read_att_values(self):
        nc_type = self.fp.read(4)
        n = self._unpack_int()

        typecode, size = TYPEMAP[nc_type]

        count = n*size
        values = self.fp.read(int(count))
        self.fp.read(-count % 4)  # read padding

        if typecode != 'c':
            values = frombuffer(values, dtype='>%s' % typecode).copy()
            if values.shape == (1,):
                values = values[0]
        else:
            values = values.rstrip(b'\x00')
        return values

    def _pack_begin(self, begin):
        if self.version_byte == 1:
            self._pack_int(begin)
        elif self.version_byte == 2:
            self._pack_int64(begin)

    def _pack_int(self, value):
        self.fp.write(array(value, '>i').tobytes())
    _pack_int32 = _pack_int

    def _unpack_int(self):
        return int(frombuffer(self.fp.read(4), '>i')[0])
    _unpack_int32 = _unpack_int

    def _pack_int64(self, value):
        self.fp.write(array(value, '>q').tobytes())

    def _unpack_int64(self):
        return frombuffer(self.fp.read(8), '>q')[0]

    def _pack_string(self, s):
        count = len(s)
        self._pack_int(count)
        self.fp.write(s.encode('latin1'))
        self.fp.write(b'\x00' * (-count % 4))  # pad

    def _unpack_string(self):
        count = self._unpack_int()
        s = self.fp.read(count).rstrip(b'\x00')
        self.fp.read(-count % 4)  # read padding
        return s


class netcdf_variable:
    """
    A data object for netcdf files.

    `netcdf_variable` objects are constructed by calling the method
    `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
    objects behave much like array objects defined in numpy, except that their
    data resides in a file. Data is read by indexing and written by assigning
    to an indexed subset; the entire array can be accessed by the index ``[:]``
    or (for scalars) by using the methods `getValue` and `assignValue`.
    `netcdf_variable` objects also have attribute `shape` with the same meaning
    as for arrays, but the shape cannot be modified. There is another read-only
    attribute `dimensions`, whose value is the tuple of dimension names.

    All other attributes correspond to variable attributes defined in
    the NetCDF file. Variable attributes are created by assigning to an
    attribute of the `netcdf_variable` object.

    Parameters
    ----------
    data : array_like
        The data array that holds the values for the variable.
        Typically, this is initialized as empty, but with the proper shape.
    typecode : dtype character code
        Desired data-type for the data array.
    size : int
        Desired element size for the data array.
    shape : sequence of ints
        The shape of the array. This should match the lengths of the
        variable's dimensions.
    dimensions : sequence of strings
        The names of the dimensions used by the variable. Must be in the
        same order of the dimension lengths given by `shape`.
    attributes : dict, optional
        Attribute values (any type) keyed by string names. These attributes
        become attributes for the netcdf_variable object.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.


    Attributes
    ----------
    dimensions : list of str
        List of names of dimensions used by the variable object.
    isrec, shape
        Properties

    See also
    --------
    isrec, shape

    """
    def __init__(self, data, typecode, size, shape, dimensions,
                 attributes=None,
                 maskandscale=False):
        self.data = data
        self._typecode = typecode
        self._size = size
        self._shape = shape
        self.dimensions = dimensions
        self.maskandscale = maskandscale

        self._attributes = attributes or {}
        for k, v in self._attributes.items():
            self.__dict__[k] = v

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def isrec(self):
        """Returns whether the variable has a record dimension or not.

        A record dimension is a dimension along which additional data could be
        easily appended in the netcdf data structure without much rewriting of
        the data file. This attribute is a read-only property of the
        `netcdf_variable`.

        """
        return bool(self.data.shape) and not self._shape[0]
    isrec = property(isrec)

    def shape(self):
        """Returns the shape tuple of the data variable.

        This is a read-only attribute and can not be modified in the
        same manner of other numpy arrays.
        """
        return self.data.shape
    shape = property(shape)

    def getValue(self):
        """
        Retrieve a scalar value from a `netcdf_variable` of length one.

        Raises
        ------
        ValueError
            If the netcdf variable is an array of length greater than one,
            this exception will be raised.

        """
        return self.data.item()

    def assignValue(self, value):
        """
        Assign a scalar value to a `netcdf_variable` of length one.

        Parameters
        ----------
        value : scalar
            Scalar value (of compatible type) to assign to a length-one netcdf
            variable. This value will be written to file.

        Raises
        ------
        ValueError
            If the input is not a scalar, or if the destination is not a length-one
            netcdf variable.

        """
        if not self.data.flags.writeable:
            # Work-around for a bug in NumPy.  Calling itemset() on a read-only
            # memory-mapped array causes a seg. fault.
            # See NumPy ticket #1622, and SciPy ticket #1202.
            # This check for `writeable` can be removed when the oldest version
            # of NumPy still supported by scipy contains the fix for #1622.
            raise RuntimeError("variable is not writeable")

        self.data.itemset(value)

    def typecode(self):
        """
        Return the typecode of the variable.

        Returns
        -------
        typecode : char
            The character typecode of the variable (e.g., 'i' for int).

        """
        return self._typecode

    def itemsize(self):
        """
        Return the itemsize of the variable.

        Returns
        -------
        itemsize : int
            The element size of the variable (e.g., 8 for float64).

        """
        return self._size

    def __getitem__(self, index):
        if not self.maskandscale:
            return self.data[index]

        data = self.data[index].copy()
        missing_value = self._get_missing_value()
        data = self._apply_missing_value(data, missing_value)
        scale_factor = self._attributes.get('scale_factor')
        add_offset = self._attributes.get('add_offset')
        if add_offset is not None or scale_factor is not None:
            data = data.astype(np.float64)
        if scale_factor is not None:
            data = data * scale_factor
        if add_offset is not None:
            data += add_offset

        return data

    def __setitem__(self, index, data):
        if self.maskandscale:
            missing_value = (
                    self._get_missing_value() or
                    getattr(data, 'fill_value', 999999))
            self._attributes.setdefault('missing_value', missing_value)
            self._attributes.setdefault('_FillValue', missing_value)
            data = ((data - self._attributes.get('add_offset', 0.0)) /
                    self._attributes.get('scale_factor', 1.0))
            data = np.ma.asarray(data).filled(missing_value)
            if self._typecode not in 'fd' and data.dtype.kind == 'f':
                data = np.round(data)

        # Expand data for record vars?
        if self.isrec:
            if isinstance(index, tuple):
                rec_index = index[0]
            else:
                rec_index = index
            if isinstance(rec_index, slice):
                recs = (rec_index.start or 0) + len(data)
            else:
                recs = rec_index + 1
            if recs > len(self.data):
                shape = (recs,) + self._shape[1:]
                # Resize in-place does not always work since
                # the array might not be single-segment
                try:
                    self.data.resize(shape)
                except ValueError:
                    self.__dict__['data'] = np.resize(self.data, shape).astype(self.data.dtype)
        self.data[index] = data

    def _default_encoded_fill_value(self):
        """
        The default encoded fill-value for this Variable's data type.
        """
        nc_type = REVERSE[self.typecode(), self.itemsize()]
        return FILLMAP[nc_type]

    def _get_encoded_fill_value(self):
        """
        Returns the encoded fill value for this variable as bytes.

        This is taken from either the _FillValue attribute, or the default fill
        value for this variable's data type.
        """
        if '_FillValue' in self._attributes:
            fill_value = np.array(self._attributes['_FillValue'],
                                  dtype=self.data.dtype).tobytes()
            if len(fill_value) == self.itemsize():
                return fill_value
            else:
                return self._default_encoded_fill_value()
        else:
            return self._default_encoded_fill_value()

    def _get_missing_value(self):
        """
        Returns the value denoting "no data" for this variable.

        If this variable does not have a missing/fill value, returns None.

        If both _FillValue and missing_value are given, give precedence to
        _FillValue. The netCDF standard gives special meaning to _FillValue;
        missing_value is  just used for compatibility with old datasets.
        """

        if '_FillValue' in self._attributes:
            missing_value = self._attributes['_FillValue']
        elif 'missing_value' in self._attributes:
            missing_value = self._attributes['missing_value']
        else:
            missing_value = None

        return missing_value

    @staticmethod
    def _apply_missing_value(data, missing_value):
        """
        Applies the given missing value to the data array.

        Returns a numpy.ma array, with any value equal to missing_value masked
        out (unless missing_value is None, in which case the original array is
        returned).
        """

        if missing_value is None:
            newdata = data
        else:
            try:
                missing_value_isnan = np.isnan(missing_value)
            except (TypeError, NotImplementedError):
                # some data types (e.g., characters) cannot be tested for NaN
                missing_value_isnan = False

            if missing_value_isnan:
                mymask = np.isnan(data)
            else:
                mymask = (data == missing_value)

            newdata = np.ma.masked_where(mymask, data)

        return newdata


NetCDFFile = netcdf_file
NetCDFVariable = netcdf_variable
